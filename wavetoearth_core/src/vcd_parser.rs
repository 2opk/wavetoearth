use std::fs::File;
use std::io;
use std::sync::Arc;
use std::collections::HashMap;
use rustc_hash::FxHashMap;
use memmap2::Mmap;
use memchr::memchr;
use arrow::array::{UInt64Builder, StringBuilder, ArrayBuilder, UInt32Builder, DictionaryArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::prelude::*;

pub struct FastVcdParser {
    mmap: Mmap,
}

impl FastVcdParser {
    pub fn new(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn parse_to_parquet(&self, parquet_path: &str, chunk_size: usize) -> pyo3::PyResult<()> {
        // Define Schema: signal_name is Dictionary(UInt32, Utf8)
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_name", DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)), false),
            Field::new("value", DataType::Utf8, false),
        ]);
        let schema = Arc::new(schema);

        let file = File::create(parquet_path)?;
        let props = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Parquet init error: {}", e)))?;

        // Builders
        let mut ts_builder = UInt64Builder::with_capacity(chunk_size);
        let mut name_indices_builder = UInt32Builder::with_capacity(chunk_size);
        let mut val_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 2);

        // Parsing State
        let mut current_time: u64 = 0;

        // Dictionary Maps
        let mut id_to_index: FxHashMap<&[u8], u32> = FxHashMap::default();
        let mut names_vec: Vec<String> = Vec::new();

        // 1. Header Parsing
        let mut cursor = 0;
        let data = &self.mmap[..];
        let len = data.len();

        while cursor < len {
             let end = memchr(b'\n', &data[cursor..]).map(|i| cursor + i).unwrap_or(len);
             let line = &data[cursor..end];

             if line.is_empty() {
                 cursor = end + 1;
                 continue;
             }

             if line[0] == b'$' {
                 if line.starts_with(b"$enddefinitions") {
                     cursor = end + 1;
                     break;
                 }
                 if line.len() > 5 && line.starts_with(b"$var") {
                     let tokens: Vec<&[u8]> = line.split(|&b| b == b' ' || b == b'\t').filter(|s| !s.is_empty()).collect();
                     if tokens.len() >= 5 {
                         let id = tokens[3];
                         let name = unsafe { std::str::from_utf8_unchecked(tokens[4]) }.to_string();

                         if !id_to_index.contains_key(id) {
                             let idx = names_vec.len() as u32;
                             id_to_index.insert(id, idx);
                             names_vec.push(name);
                         }
                     }
                 }
             }
             cursor = end + 1;
        }

        // Pre-create the Dictionary Values Array
        let dictionary_values = Arc::new(StringArray::from(names_vec));

        // 2. Body Parsing (Optimized Loop)
        while cursor < len {
            let end = memchr(b'\n', &data[cursor..]).map(|i| cursor + i).unwrap_or(len);
            let line = &data[cursor..end];
            if line.is_empty() { cursor = end + 1; continue; }

            // Inline match for performance
            let first_byte = line[0];
            if first_byte == b'#' {
                 let time_str = unsafe { std::str::from_utf8_unchecked(&line[1..]) };
                 if let Ok(t) = time_str.parse::<u64>() {
                     current_time = t;
                 }
            } else if first_byte == b'0' || first_byte == b'1' || first_byte == b'x' || first_byte == b'z' {
                 let id = &line[1..];
                 if let Some(&idx) = id_to_index.get(id) {
                     ts_builder.append_value(current_time);
                     name_indices_builder.append_value(idx);
                     let s = unsafe { std::str::from_utf8_unchecked(&line[0..1]) };
                     val_builder.append_value(s);
                 }
            } else if first_byte == b'b' || first_byte == b'B' {
                 if let Some(space_idx) = memchr(b' ', line) {
                     let value = &line[1..space_idx];
                     let id = &line[space_idx+1..];
                     if let Some(&idx) = id_to_index.get(id) {
                         ts_builder.append_value(current_time);
                         name_indices_builder.append_value(idx);
                         val_builder.append_value(unsafe { std::str::from_utf8_unchecked(value) });
                     }
                 }
            }

            if ts_builder.len() >= chunk_size {
                 let keys = name_indices_builder.finish();
                 // Create Dictionary Array
                 // keys: UInt32Array, values: StringArray
                 // Safety: keys are guaranteed valid indices provided by logic
                 let dict_array = DictionaryArray::<UInt32Type>::try_new(keys, dictionary_values.clone())
                     .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dict Error: {}", e)))?;

                 let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(ts_builder.finish()), // ts_builder reset
                        Arc::new(dict_array),
                        Arc::new(val_builder.finish()),
                    ]
                ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
                writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
            }

            cursor = end + 1;
        }

       // Final Flush
        if ts_builder.len() > 0 {
             let keys = name_indices_builder.finish();
             let dict_array = DictionaryArray::<UInt32Type>::try_new(keys, dictionary_values.clone())
                 .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dict Error: {}", e)))?;

             let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(ts_builder.finish()),
                    Arc::new(dict_array),
                    Arc::new(val_builder.finish()),
                ]
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
            writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
        }
        writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Close error: {}", e)))?;

        Ok(())
    }
}
