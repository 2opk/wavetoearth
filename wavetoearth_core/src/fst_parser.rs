use std::fs::File;
use std::io;
use std::sync::Arc;
use rustc_hash::FxHashMap;
use arrow::array::{UInt64Builder, StringBuilder, ArrayBuilder, UInt32Builder, DictionaryArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::prelude::*;
use fst_native::{FstReader, FstFilter, FstSignalHandle, FstSignalValue, FstHierarchyEntry};

pub struct FastFstParser {
    path: String,
}

impl FastFstParser {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }

    pub fn parse_to_parquet(&self, parquet_path: &str, chunk_size: usize) -> pyo3::PyResult<()> {
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_name", DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)), false),
            Field::new("value", DataType::Utf8, false),
        ]);
        let schema = Arc::new(schema);

        // 1. Open FST Reader
        let file = File::open(&self.path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("FST Open Error: {}", e)))?;
        let mut reader = FstReader::open(std::io::BufReader::new(file)).unwrap(); // TODO: Handle error properly

        // 2. Build Hierarchy (Handle -> Name Index)
        let mut handle_to_index: FxHashMap<u64, u32> = FxHashMap::default();
        let mut names_vec: Vec<String> = Vec::new();

        // We need to reconstruct full names from hierarchy
        // This is tricky with callback.
        // We can maintain a stack of scope names.
        // Or simpler: Just use what we get.
        // fst-native hierarchy callback supplies entry.
        // Scope push/pop.

        let mut scope_stack: Vec<String> = Vec::new();
        // Temporary map to build full names

        let hierarchy_cb = |entry: FstHierarchyEntry| {
            match entry {
                FstHierarchyEntry::Scope { name, .. } => {
                    scope_stack.push(name.to_string());
                },
                FstHierarchyEntry::UpScope => {
                    scope_stack.pop();
                },
                FstHierarchyEntry::Var { name, handle, .. } => {
                    // Full name = scope_stack + name
                    let full_name = if scope_stack.is_empty() {
                        name.to_string()
                    } else {
                        format!("{}.{}", scope_stack.join("."), name)
                    };

                    let handle_idx = handle.get_index() as u64;

                    if !handle_to_index.contains_key(&handle_idx) {
                        let idx = names_vec.len() as u32;
                        handle_to_index.insert(handle_idx, idx);
                        names_vec.push(full_name);
                    }
                },
                _ => {}
            }
        };

        reader.read_hierarchy(hierarchy_cb).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Hierarchy Error: {}", e)))?;

        let dictionary_values = Arc::new(StringArray::from(names_vec));

        // 3. Parquet Setup
        let file_out = File::create(parquet_path)?;
        let props = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
        let mut writer = ArrowWriter::try_new(file_out, schema.clone(), Some(props))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Parquet init error: {}", e)))?;

        // 4. Signal Iteration
        let mut ts_builder = UInt64Builder::with_capacity(chunk_size);
        let mut name_indices_builder = UInt32Builder::with_capacity(chunk_size);
        let mut val_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 2);

        let filter = FstFilter::all(); // Read all signals

        let mut flush_batch = |ts: &mut UInt64Builder, name: &mut UInt32Builder, val: &mut StringBuilder| -> PyResult<()> {
            if ts.len() == 0 { return Ok(()); }
            let keys = name.finish();
            let dict_array = DictionaryArray::<UInt32Type>::try_new(keys, dictionary_values.clone())
                     .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dict Error: {}", e)))?;

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(ts.finish()),
                    Arc::new(dict_array),
                    Arc::new(val.finish()),
                ]
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
            writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
            Ok(())
        };

        let signal_cb = |time: u64, handle: FstSignalHandle, value: FstSignalValue| {
             let handle_idx = handle.get_index() as u64;
             if let Some(&idx) = handle_to_index.get(&handle_idx) {
                 ts_builder.append_value(time);
                 name_indices_builder.append_value(idx);

                 match value {
                     FstSignalValue::String(bytes) => {
                         let s = String::from_utf8_lossy(bytes);
                         val_builder.append_value(s);
                     },
                     FstSignalValue::Real(v) => {
                         val_builder.append_value(v.to_string());
                     }
                 }

                 if ts_builder.len() >= chunk_size {
                     // Flush
                     // Cannot return Result from closure easily if method doesn't support it?
                     // fst-native callbacks return () usually.
                     // We panic on error? Or store error in external var?
                     // For now, simple panic or unwrap.
                     // But strictly we should handle it.
                     // The compiler might complain about `?` in closure returning `()`.
                     // Fix: use unwrap for now.
                     flush_batch(&mut ts_builder, &mut name_indices_builder, &mut val_builder).unwrap();
                 }
             }
        };

        reader.read_signals(&filter, signal_cb).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Read Signals Error: {}", e)))?;

        // Final Flush
        if ts_builder.len() > 0 {
            flush_batch(&mut ts_builder, &mut name_indices_builder, &mut val_builder)?;
        }
        writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Close error: {}", e)))?;

        Ok(())
    }
}
