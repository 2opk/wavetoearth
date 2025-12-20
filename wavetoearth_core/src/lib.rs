use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::process::{Command as SysCommand, Stdio};
use std::sync::Arc;
use vcd::{Parser, Command};
use wellen::{GetItem, SignalValue};
use wellen::simple;
use arrow::array::{UInt64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

mod vcd_parser;
use vcd_parser::FastVcdParser;

mod fst_parser;
use fst_parser::FastFstParser;

#[pyclass]
struct WaveParser {}

#[pymethods]
impl WaveParser {
    #[new]
    fn new() -> Self {
        WaveParser {}
    }

    /// Converts a VCD file to a Parquet file using FastVCD (Zero-Copy, SIMD)
    fn convert_to_parquet(&self, vcd_path: String, parquet_path: String, chunk_size: usize) -> PyResult<()> {
        let parser = FastVcdParser::new(&vcd_path).map_err(|e| {
             PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("FastVCD Init Error: {}", e))
        })?;
        parser.parse_to_parquet(&parquet_path, chunk_size)?;
        Ok(())
    }

    /// Converts an FST file to a Parquet file using FastFST (Native Streaming)
    fn convert_fst_to_parquet(&self, fst_path: String, parquet_path: String, chunk_size: usize) -> PyResult<()> {
        let parser = FastFstParser::new(&fst_path);
        parser.parse_to_parquet(&parquet_path, chunk_size)?;
        Ok(())
    }
}

impl WaveParser {
    fn run_vcd_parser<R: std::io::Read>(reader: BufReader<R>, parquet_path: String, chunk_size: usize) -> PyResult<()> {
        let mut parser = Parser::new(reader);

        // Header Parsing
        let header = parser.parse_header().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Header error: {}", e))
        })?;

        let mut id_to_name: HashMap<vcd::IdCode, String> = HashMap::new();
        let mut scope_stack: Vec<String> = Vec::new();

        fn walk(items: &[vcd::ScopeItem], stack: &mut Vec<String>, map: &mut HashMap<vcd::IdCode, String>) {
            for item in items {
                match item {
                    vcd::ScopeItem::Var(var) => {
                         let name = var.reference.clone();
                         let full_name = if stack.is_empty() { name } else { format!("{}.{}", stack.join("."), name) };
                         map.insert(var.code, full_name);
                    },
                    vcd::ScopeItem::Scope(scope) => {
                        stack.push(scope.identifier.clone());
                        walk(&scope.children, stack, map);
                        stack.pop();
                    },
                    _ => {}
                }
            }
        }
        walk(&header.items, &mut scope_stack, &mut id_to_name);

        // Parquet Writer Init
        let file = File::create(&parquet_path)?;
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_name", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]);
        let schema = Arc::new(schema);
        let props = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Parquet init error: {}", e)))?;

        // Buffers
        let mut ts_builder = UInt64Builder::with_capacity(chunk_size);
        let mut name_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 10);
        let mut val_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 2);
        let mut row_count = 0;
        let mut current_time = 0;

        for result in parser {
            let command = result.map_err(|e| {
                 PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Parse error: {}", e))
            })?;

            match command {
                Command::Timestamp(t) => current_time = t,
                Command::ChangeScalar(id, val) => {
                    if let Some(name) = id_to_name.get(&id) {
                        ts_builder.append_value(current_time);
                        name_builder.append_value(name);
                        val_builder.append_value(val.to_string());
                        row_count += 1;
                    }
                },
                Command::ChangeVector(id, val) => {
                     if let Some(name) = id_to_name.get(&id) {
                        ts_builder.append_value(current_time);
                        name_builder.append_value(name);
                         let s: String = val.iter().map(|b| b.to_string()).collect();
                        val_builder.append_value(s);
                        row_count += 1;
                    }
                }
                _ => {}
            }

            if row_count >= chunk_size {
                 let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(ts_builder.finish()),
                        Arc::new(name_builder.finish()),
                        Arc::new(val_builder.finish()),
                    ]
                ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
                writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;

                ts_builder = UInt64Builder::with_capacity(chunk_size);
                name_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 10);
                val_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 2);
                row_count = 0;
            }
        }
        if row_count > 0 {
             let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(ts_builder.finish()),
                    Arc::new(name_builder.finish()),
                    Arc::new(val_builder.finish()),
                ]
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
            writer.write(&batch).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
        }
        writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Close error: {}", e)))?;
        Ok(())
    }
}


#[pymodule]
fn wavetoearth_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WaveParser>()?;
    Ok(())
}
