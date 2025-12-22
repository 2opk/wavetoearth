use std::fs::File;
use std::sync::Arc;

use arrow::array::{ArrayBuilder, BooleanBuilder, Float64Builder, Int32Builder, StringBuilder, UInt32Builder, UInt64Builder, UInt8Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use fst_native::{FstFilter, FstHierarchyEntry, FstReader, FstSignalHandle, FstSignalValue};

pub struct FastFstParser {
    path: String,
}

struct FstSignalMeta {
    signal_id: u32,
    full_name: String,
    handle: u64,
    var_type: String,
    direction: String,
    length: u32,
    is_alias: bool,
}

struct FstGlobalMeta {
    version: String,
    date: String,
    timescale_exponent: i32,
    timescale_seconds: f64,
    start_time: u64,
    end_time: u64,
    var_count: u64,
    max_handle: u64,
}

impl FastFstParser {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }

    pub fn parse_to_parquet(&self, parquet_path: &str, chunk_size: usize) -> PyResult<()> {
        let (handle_to_index, meta, global) = self.read_hierarchy()?;
        let meta_path = format!("{}.meta.parquet", parquet_path);
        let global_path = format!("{}.global.parquet", parquet_path);
        write_fst_meta_parquet(&meta_path, &meta)?;
        write_global_meta_parquet(
            &global_path,
            "fst",
            Some(global.version.as_str()),
            Some(global.date.as_str()),
            None,
            Some(global.timescale_seconds),
            Some(global.timescale_exponent),
            Some(global.start_time),
            Some(global.end_time),
            Some(global.var_count),
            Some(global.max_handle),
        )?;
        self.write_shard(
            parquet_path,
            chunk_size,
            &handle_to_index,
            None,
        )
    }

    pub fn parse_to_parquet_sharded(
        &self,
        parquet_dir: &str,
        chunk_size: usize,
        shards: usize,
    ) -> PyResult<()> {
        let (handle_to_index, meta, global) = self.read_hierarchy()?;
        let meta_path = format!("{}/_meta.parquet", parquet_dir);
        let global_path = format!("{}/_global.parquet", parquet_dir);
        write_fst_meta_parquet(&meta_path, &meta)?;
        write_global_meta_parquet(
            &global_path,
            "fst",
            Some(global.version.as_str()),
            Some(global.date.as_str()),
            None,
            Some(global.timescale_seconds),
            Some(global.timescale_exponent),
            Some(global.start_time),
            Some(global.end_time),
            Some(global.var_count),
            Some(global.max_handle),
        )?;
        let shard_count = if shards == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            shards
        }
        .max(1);

        std::fs::create_dir_all(parquet_dir).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Parquet dir create error: {}",
                e
            ))
        })?;

        self.write_shards_single_pass(
            parquet_dir,
            chunk_size,
            shard_count,
            &handle_to_index,
            global.max_handle,
        )
    }

    fn read_hierarchy(&self) -> PyResult<(FxHashMap<u64, u32>, Vec<FstSignalMeta>, FstGlobalMeta)> {
        let file = File::open(&self.path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("FST Open Error: {}", e)))?;
        let mut reader = FstReader::open(std::io::BufReader::new(file))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FST Open Error: {}", e)))?;

        let header = reader.get_header();
        let global = FstGlobalMeta {
            version: header.version.clone(),
            date: header.date.clone(),
            timescale_exponent: header.timescale_exponent as i32,
            timescale_seconds: 10f64.powi(header.timescale_exponent as i32),
            start_time: header.start_time,
            end_time: header.end_time,
            var_count: header.var_count,
            max_handle: header.max_handle,
        };

        let mut handle_to_index: FxHashMap<u64, u32> = FxHashMap::default();
        let mut scope_stack: Vec<String> = Vec::new();
        let mut signal_meta: Vec<FstSignalMeta> = Vec::new();

        let hierarchy_cb = |entry: FstHierarchyEntry| {
            match entry {
                FstHierarchyEntry::Scope { name, .. } => {
                    scope_stack.push(name.to_string());
                }
                FstHierarchyEntry::UpScope => {
                    scope_stack.pop();
                }
                FstHierarchyEntry::Var {
                    name,
                    handle,
                    tpe,
                    direction,
                    length,
                    is_alias,
                } => {
                    let full_name = if scope_stack.is_empty() {
                        name.to_string()
                    } else {
                        format!("{}.{}", scope_stack.join("."), name)
                    };
                    let handle_idx = handle.get_index() as u64;
                    let signal_id = if let Some(&idx) = handle_to_index.get(&handle_idx) {
                        idx
                    } else {
                        let idx = handle_to_index.len() as u32;
                        handle_to_index.insert(handle_idx, idx);
                        idx
                    };
                    signal_meta.push(FstSignalMeta {
                        signal_id,
                        full_name,
                        handle: handle_idx,
                        var_type: format!("{:?}", tpe),
                        direction: format!("{:?}", direction),
                        length,
                        is_alias,
                    });
                }
                _ => {}
            }
        };

        reader
            .read_hierarchy(hierarchy_cb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Hierarchy Error: {}", e)))?;

        Ok((handle_to_index, signal_meta, global))
    }

    fn write_shard(
        &self,
        parquet_path: &str,
        chunk_size: usize,
        handle_to_index: &FxHashMap<u64, u32>,
        filter: Option<FstFilter>,
    ) -> PyResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_id", DataType::UInt32, false),
            Field::new("value_raw", DataType::Utf8, false),
            Field::new("value_kind", DataType::UInt8, false),
            Field::new("value_scalar", DataType::UInt8, true),
            Field::new("value_real", DataType::Float64, true),
        ]));

        let file = File::create(parquet_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Parquet init error: {}", e)))?;

        let mut ts_builder = UInt64Builder::with_capacity(chunk_size);
        let mut id_builder = UInt32Builder::with_capacity(chunk_size);
        let mut raw_builder = StringBuilder::with_capacity(chunk_size, chunk_size * 4);
        let mut kind_builder = UInt8Builder::with_capacity(chunk_size);
        let mut scalar_builder = UInt8Builder::with_capacity(chunk_size);
        let mut real_builder = Float64Builder::with_capacity(chunk_size);

        let file_in = File::open(&self.path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("FST Open Error: {}", e)))?;
        let mut reader = FstReader::open(std::io::BufReader::new(file_in))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FST Open Error: {}", e)))?;

        let filter = filter.unwrap_or_else(FstFilter::all);
        let mut flush_batch = |ts: &mut UInt64Builder,
                               id: &mut UInt32Builder,
                               raw: &mut StringBuilder,
                               kind: &mut UInt8Builder,
                               scalar: &mut UInt8Builder,
                               real: &mut Float64Builder|
         -> PyResult<()> {
            if ts.len() == 0 {
                return Ok(());
            }
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(ts.finish()),
                    Arc::new(id.finish()),
                    Arc::new(raw.finish()),
                    Arc::new(kind.finish()),
                    Arc::new(scalar.finish()),
                    Arc::new(real.finish()),
                ],
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
            writer
                .write(&batch)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
            Ok(())
        };

        let mut error: Option<PyErr> = None;
        let signal_cb = |time: u64, handle: FstSignalHandle, value: FstSignalValue| {
            if error.is_some() {
                return;
            }
            let handle_idx = handle.get_index() as u64;
            if let Some(&idx) = handle_to_index.get(&handle_idx) {
                ts_builder.append_value(time);
                id_builder.append_value(idx);

                match value {
                    FstSignalValue::String(bytes) => {
                        let s = unsafe { std::str::from_utf8_unchecked(bytes) };
                        raw_builder.append_value(s);
                        if s.len() == 1 {
                            kind_builder.append_value(0);
                            scalar_builder.append_value(map_scalar(s.as_bytes()[0]));
                            real_builder.append_null();
                        } else {
                            kind_builder.append_value(1);
                            scalar_builder.append_null();
                            real_builder.append_null();
                        }
                    }
                    FstSignalValue::Real(v) => {
                        raw_builder.append_value(v.to_string());
                        kind_builder.append_value(2);
                        scalar_builder.append_null();
                        real_builder.append_value(v);
                    }
                }

                if ts_builder.len() >= chunk_size {
                    if let Err(e) = flush_batch(
                        &mut ts_builder,
                        &mut id_builder,
                        &mut raw_builder,
                        &mut kind_builder,
                        &mut scalar_builder,
                        &mut real_builder,
                    ) {
                        error = Some(e);
                    }
                }
            }
        };

        reader
            .read_signals(&filter, signal_cb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Read Signals Error: {}", e)))?;

        if let Some(e) = error {
            return Err(e);
        }

        if ts_builder.len() > 0 {
            flush_batch(
                &mut ts_builder,
                &mut id_builder,
                &mut raw_builder,
                &mut kind_builder,
                &mut scalar_builder,
                &mut real_builder,
            )?;
        }
        writer.close().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Close error: {}", e)))?;
        Ok(())
    }

    fn write_shards_single_pass(
        &self,
        parquet_dir: &str,
        chunk_size: usize,
        shard_count: usize,
        handle_to_index: &FxHashMap<u64, u32>,
        max_handle: u64,
    ) -> PyResult<()> {
        struct ShardState {
            writer: Option<ArrowWriter<File>>,
            ts_builder: UInt64Builder,
            id_builder: UInt32Builder,
            raw_builder: StringBuilder,
            kind_builder: UInt8Builder,
            scalar_builder: UInt8Builder,
            real_builder: Float64Builder,
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_id", DataType::UInt32, false),
            Field::new("value_raw", DataType::Utf8, false),
            Field::new("value_kind", DataType::UInt8, false),
            Field::new("value_scalar", DataType::UInt8, true),
            Field::new("value_real", DataType::Float64, true),
        ]));

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let mut shards: Vec<ShardState> = Vec::with_capacity(shard_count);
        for i in 0..shard_count {
            let parquet_path = format!("{}/part-{:05}.parquet", parquet_dir, i);
            let file = File::create(&parquet_path)?;
            let writer = ArrowWriter::try_new(file, schema.clone(), Some(props.clone()))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Parquet init error: {}", e)))?;
            shards.push(ShardState {
                writer: Some(writer),
                ts_builder: UInt64Builder::with_capacity(chunk_size),
                id_builder: UInt32Builder::with_capacity(chunk_size),
                raw_builder: StringBuilder::with_capacity(chunk_size, chunk_size * 4),
                kind_builder: UInt8Builder::with_capacity(chunk_size),
                scalar_builder: UInt8Builder::with_capacity(chunk_size),
                real_builder: Float64Builder::with_capacity(chunk_size),
            });
        }

        let max_handle = max_handle as usize;
        let mut handle_to_signal: Vec<u32> = vec![u32::MAX; max_handle + 1];
        for (handle, idx) in handle_to_index {
            let h = *handle as usize;
            if h <= max_handle {
                handle_to_signal[h] = *idx;
            }
        }

        let signals_per_shard = ((handle_to_index.len() + shard_count - 1) / shard_count).max(1);
        let mut handle_to_shard: Vec<usize> = vec![0; max_handle + 1];
        for h in 0..=max_handle {
            let mut shard = h / signals_per_shard;
            if shard >= shard_count {
                shard = shard_count - 1;
            }
            handle_to_shard[h] = shard;
        }

        let flush_shard = |shard: &mut ShardState| -> PyResult<()> {
            if shard.ts_builder.len() == 0 {
                return Ok(());
            }
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(shard.ts_builder.finish()),
                    Arc::new(shard.id_builder.finish()),
                    Arc::new(shard.raw_builder.finish()),
                    Arc::new(shard.kind_builder.finish()),
                    Arc::new(shard.scalar_builder.finish()),
                    Arc::new(shard.real_builder.finish()),
                ],
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
            if let Some(writer) = shard.writer.as_mut() {
                writer
                    .write(&batch)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
            }
            Ok(())
        };

        let file_in = File::open(&self.path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("FST Open Error: {}", e)))?;
        let mut reader = FstReader::open(std::io::BufReader::new(file_in))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FST Open Error: {}", e)))?;

        let filter = FstFilter::all();
        let mut error: Option<PyErr> = None;
        let signal_cb = |time: u64, handle: FstSignalHandle, value: FstSignalValue| {
            if error.is_some() {
                return;
            }
            let handle_idx = handle.get_index();
            if handle_idx > max_handle {
                return;
            }
            let signal_id = handle_to_signal[handle_idx];
            if signal_id == u32::MAX {
                return;
            }
            let shard_idx = handle_to_shard[handle_idx];
            let shard = &mut shards[shard_idx];

            shard.ts_builder.append_value(time);
            shard.id_builder.append_value(signal_id);

            match value {
                FstSignalValue::String(bytes) => {
                    let s = unsafe { std::str::from_utf8_unchecked(bytes) };
                    shard.raw_builder.append_value(s);
                    if s.len() == 1 {
                        shard.kind_builder.append_value(0);
                        shard.scalar_builder.append_value(map_scalar(s.as_bytes()[0]));
                        shard.real_builder.append_null();
                    } else {
                        shard.kind_builder.append_value(1);
                        shard.scalar_builder.append_null();
                        shard.real_builder.append_null();
                    }
                }
                FstSignalValue::Real(v) => {
                    shard.raw_builder.append_value(v.to_string());
                    shard.kind_builder.append_value(2);
                    shard.scalar_builder.append_null();
                    shard.real_builder.append_value(v);
                }
            }

            if shard.ts_builder.len() >= chunk_size {
                if let Err(e) = flush_shard(shard) {
                    error = Some(e);
                }
            }
        };

        reader
            .read_signals(&filter, signal_cb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Read Signals Error: {}", e)))?;

        if let Some(e) = error {
            return Err(e);
        }

        for shard in shards.iter_mut() {
            flush_shard(shard)?;
            if let Some(writer) = shard.writer.take() {
                writer
                    .close()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Close error: {}", e)))?;
            }
        }

        Ok(())
    }
}

fn write_fst_meta_parquet(path: &str, signals: &[FstSignalMeta]) -> PyResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("signal_id", DataType::UInt32, false),
        Field::new("signal_name", DataType::Utf8, false),
        Field::new("handle", DataType::UInt64, false),
        Field::new("var_type", DataType::Utf8, false),
        Field::new("direction", DataType::Utf8, false),
        Field::new("length", DataType::UInt32, false),
        Field::new("is_alias", DataType::Boolean, false),
    ]));

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Meta Parquet init error: {}", e)))?;

    let mut signal_id_builder = UInt32Builder::with_capacity(signals.len());
    let mut name_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 16);
    let mut handle_builder = UInt64Builder::with_capacity(signals.len());
    let mut var_type_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 8);
    let mut direction_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 8);
    let mut length_builder = UInt32Builder::with_capacity(signals.len());
    let mut alias_builder = BooleanBuilder::with_capacity(signals.len());

    for meta in signals {
        signal_id_builder.append_value(meta.signal_id);
        name_builder.append_value(&meta.full_name);
        handle_builder.append_value(meta.handle);
        var_type_builder.append_value(&meta.var_type);
        direction_builder.append_value(&meta.direction);
        length_builder.append_value(meta.length);
        alias_builder.append_value(meta.is_alias);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(signal_id_builder.finish()),
            Arc::new(name_builder.finish()),
            Arc::new(handle_builder.finish()),
            Arc::new(var_type_builder.finish()),
            Arc::new(direction_builder.finish()),
            Arc::new(length_builder.finish()),
            Arc::new(alias_builder.finish()),
        ],
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Meta batch error: {}", e)))?;
    writer
        .write(&batch)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Meta write error: {}", e)))?;
    writer
        .close()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Meta close error: {}", e)))?;
    Ok(())
}

fn write_global_meta_parquet(
    path: &str,
    source: &str,
    version: Option<&str>,
    date: Option<&str>,
    timescale_raw: Option<&str>,
    timescale_seconds: Option<f64>,
    timescale_exponent: Option<i32>,
    start_time: Option<u64>,
    end_time: Option<u64>,
    var_count: Option<u64>,
    max_handle: Option<u64>,
) -> PyResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("version", DataType::Utf8, true),
        Field::new("date", DataType::Utf8, true),
        Field::new("timescale_raw", DataType::Utf8, true),
        Field::new("timescale_seconds", DataType::Float64, true),
        Field::new("timescale_exponent", DataType::Int32, true),
        Field::new("start_time", DataType::UInt64, true),
        Field::new("end_time", DataType::UInt64, true),
        Field::new("var_count", DataType::UInt64, true),
        Field::new("max_handle", DataType::UInt64, true),
    ]));

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Global meta init error: {}", e)))?;

    let mut source_builder = StringBuilder::with_capacity(1, source.len());
    let mut version_builder = StringBuilder::with_capacity(1, 64);
    let mut date_builder = StringBuilder::with_capacity(1, 64);
    let mut timescale_raw_builder = StringBuilder::with_capacity(1, 16);
    let mut timescale_seconds_builder = Float64Builder::with_capacity(1);
    let mut timescale_exp_builder = Int32Builder::with_capacity(1);
    let mut start_builder = UInt64Builder::with_capacity(1);
    let mut end_builder = UInt64Builder::with_capacity(1);
    let mut var_count_builder = UInt64Builder::with_capacity(1);
    let mut max_handle_builder = UInt64Builder::with_capacity(1);

    source_builder.append_value(source);
    if let Some(v) = version {
        version_builder.append_value(v);
    } else {
        version_builder.append_null();
    }
    if let Some(d) = date {
        date_builder.append_value(d);
    } else {
        date_builder.append_null();
    }
    if let Some(raw) = timescale_raw {
        timescale_raw_builder.append_value(raw);
    } else {
        timescale_raw_builder.append_null();
    }
    if let Some(sec) = timescale_seconds {
        timescale_seconds_builder.append_value(sec);
    } else {
        timescale_seconds_builder.append_null();
    }
    if let Some(exp) = timescale_exponent {
        timescale_exp_builder.append_value(exp);
    } else {
        timescale_exp_builder.append_null();
    }
    if let Some(val) = start_time {
        start_builder.append_value(val);
    } else {
        start_builder.append_null();
    }
    if let Some(val) = end_time {
        end_builder.append_value(val);
    } else {
        end_builder.append_null();
    }
    if let Some(val) = var_count {
        var_count_builder.append_value(val);
    } else {
        var_count_builder.append_null();
    }
    if let Some(val) = max_handle {
        max_handle_builder.append_value(val);
    } else {
        max_handle_builder.append_null();
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(source_builder.finish()),
            Arc::new(version_builder.finish()),
            Arc::new(date_builder.finish()),
            Arc::new(timescale_raw_builder.finish()),
            Arc::new(timescale_seconds_builder.finish()),
            Arc::new(timescale_exp_builder.finish()),
            Arc::new(start_builder.finish()),
            Arc::new(end_builder.finish()),
            Arc::new(var_count_builder.finish()),
            Arc::new(max_handle_builder.finish()),
        ],
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Global meta batch error: {}", e)))?;
    writer
        .write(&batch)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Global meta write error: {}", e)))?;
    writer
        .close()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Global meta close error: {}", e)))?;
    Ok(())
}

fn map_scalar(value: u8) -> u8 {
    match value {
        b'0' => 0,
        b'1' => 1,
        b'x' | b'X' => 2,
        b'z' | b'Z' => 3,
        _ => 255,
    }
}
