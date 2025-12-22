use std::fs::File;
use std::io;
use std::sync::Arc;

use memchr::memchr;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use arrow::array::{ArrayBuilder, Float64Builder, Int32Builder, StringBuilder, UInt32Builder, UInt64Builder, UInt8Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::prelude::*;
use memmap2::Mmap;

struct VcdHeader<'a> {
    body_start: usize,
    id_to_index: FxHashMap<&'a [u8], u32>,
    id_lens_desc: Vec<usize>,
    signal_meta: Vec<VcdSignalMeta>,
    timescale_raw: Option<String>,
    timescale_seconds: Option<f64>,
    date: Option<String>,
    version: Option<String>,
}

struct VcdSignalMeta {
    signal_id: u32,
    full_name: String,
    id_code: String,
    var_type: String,
    size: u32,
    range: Option<String>,
    scope: Option<String>,
    reference: String,
}

#[derive(Copy, Clone)]
enum VcdMetaBlock {
    Date,
    Version,
}

pub struct FastVcdParser {
    mmap: Mmap,
}

impl FastVcdParser {
    pub fn new(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn parse_to_parquet(&self, parquet_path: &str, chunk_size: usize) -> PyResult<()> {
        let header = self.parse_header()?;
        let meta_path = format!("{}.meta.parquet", parquet_path);
        let global_path = format!("{}.global.parquet", parquet_path);
        write_vcd_meta_parquet(&meta_path, &header.signal_meta)?;
        write_global_meta_parquet(
            &global_path,
            "vcd",
            header.version.as_deref(),
            header.date.as_deref(),
            header.timescale_raw.as_deref(),
            header.timescale_seconds,
            None,
            None,
            None,
            None,
            None,
        )?;
        self.parse_range_to_parquet(
            &header,
            parquet_path,
            chunk_size,
            header.body_start,
            self.mmap.len(),
        )
    }

    pub fn parse_to_parquet_sharded(
        &self,
        parquet_dir: &str,
        chunk_size: usize,
        shards: usize,
    ) -> PyResult<()> {
        let header = self.parse_header()?;
        let meta_path = format!("{}/_meta.parquet", parquet_dir);
        let global_path = format!("{}/_global.parquet", parquet_dir);
        write_vcd_meta_parquet(&meta_path, &header.signal_meta)?;
        write_global_meta_parquet(
            &global_path,
            "vcd",
            header.version.as_deref(),
            header.date.as_deref(),
            header.timescale_raw.as_deref(),
            header.timescale_seconds,
            None,
            None,
            None,
            None,
            None,
        )?;
        let data = &self.mmap[..];
        let len = data.len();

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

        let mut boundaries: Vec<usize> = Vec::with_capacity(shard_count + 1);
        boundaries.push(header.body_start);
        let total = len.saturating_sub(header.body_start);
        let approx = if shard_count > 0 {
            (total + shard_count - 1) / shard_count
        } else {
            total
        };

        for i in 1..shard_count {
            let raw = header.body_start + i * approx;
            if raw >= len {
                break;
            }
            let next_ts = find_next_timestamp_line(data, raw, len);
            if next_ts <= *boundaries.last().unwrap() || next_ts >= len {
                continue;
            }
            boundaries.push(next_ts);
        }
        boundaries.push(len);

        boundaries.sort_unstable();
        boundaries.dedup();

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_id", DataType::UInt32, false),
            Field::new("value_raw", DataType::Utf8, false),
            Field::new("value_kind", DataType::UInt8, false),
            Field::new("value_scalar", DataType::UInt8, true),
            Field::new("value_real", DataType::Float64, true),
        ]));
        let id_to_index = Arc::new(header.id_to_index);
        let id_lens_desc = Arc::new(header.id_lens_desc);

        let results: Result<Vec<_>, String> = boundaries
            .windows(2)
            .collect::<Vec<_>>()
            .into_par_iter()
            .enumerate()
            .map(|(i, window)| {
                let start = window[0];
                let end = window[1];
                if start >= end {
                    return Ok(());
                }
                let parquet_path = format!("{}/part-{:05}.parquet", parquet_dir, i);
                self.parse_range_to_parquet_with_schema(
                    data,
                    &schema,
                    &id_to_index,
                    &id_lens_desc,
                    &parquet_path,
                    chunk_size,
                    start,
                    end,
                )
                .map_err(|e| format!("Shard {} error: {}", i, e))
            })
            .collect();

        match results {
            Ok(_) => Ok(()),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err)),
        }
    }

    fn parse_header(&self) -> PyResult<VcdHeader<'_>> {
        let data = &self.mmap[..];
        let len = data.len();
        let mut cursor = 0;

        let mut scope_stack: Vec<String> = Vec::new();
        let mut id_to_index: FxHashMap<&[u8], u32> = FxHashMap::default();
        let mut id_lens: Vec<usize> = Vec::new();
        let mut signal_meta: Vec<VcdSignalMeta> = Vec::new();
        let mut timescale_raw: Option<String> = None;
        let mut timescale_seconds: Option<f64> = None;
        let mut date: Option<String> = None;
        let mut version: Option<String> = None;
        let mut timescale_pending = false;
        let mut pending_block: Option<VcdMetaBlock> = None;
        let mut pending_buf = String::new();

        while cursor < len {
            let end = memchr(b'\n', &data[cursor..]).map(|i| cursor + i).unwrap_or(len);
            let mut line = &data[cursor..end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }

            if line.is_empty() {
                cursor = end + 1;
                continue;
            }

            if let Some(block) = pending_block {
                if line.starts_with(b"$end") {
                    let value = pending_buf.trim().to_string();
                    match block {
                        VcdMetaBlock::Date => {
                            if !value.is_empty() {
                                date = Some(value);
                            }
                        }
                        VcdMetaBlock::Version => {
                            if !value.is_empty() {
                                version = Some(value);
                            }
                        }
                    }
                    pending_block = None;
                    pending_buf.clear();
                } else {
                    if !pending_buf.is_empty() {
                        pending_buf.push(' ');
                    }
                    pending_buf.push_str(unsafe { std::str::from_utf8_unchecked(line) });
                }
                cursor = end + 1;
                continue;
            }

            if line[0] == b'$' {
                if line.starts_with(b"$enddefinitions") {
                    cursor = end + 1;
                    break;
                }
                if line.starts_with(b"$timescale") {
                    if let Some(value) = extract_inline_value(line, b"$timescale") {
                        timescale_raw = Some(value);
                        timescale_seconds = timescale_raw
                            .as_deref()
                            .and_then(parse_timescale_seconds);
                        timescale_pending = false;
                    } else {
                        timescale_pending = true;
                    }
                    cursor = end + 1;
                    continue;
                }
                if line.starts_with(b"$date") {
                    if let Some(value) = extract_inline_value(line, b"$date") {
                        date = Some(value);
                    } else {
                        pending_block = Some(VcdMetaBlock::Date);
                    }
                    cursor = end + 1;
                    continue;
                }
                if line.starts_with(b"$version") {
                    if let Some(value) = extract_inline_value(line, b"$version") {
                        version = Some(value);
                    } else {
                        pending_block = Some(VcdMetaBlock::Version);
                    }
                    cursor = end + 1;
                    continue;
                }
                if line.starts_with(b"$scope") {
                    let tokens: Vec<&[u8]> = line
                        .split(|&b| b == b' ' || b == b'\t')
                        .filter(|s| !s.is_empty())
                        .collect();
                    if tokens.len() >= 3 {
                        let scope = unsafe { std::str::from_utf8_unchecked(tokens[2]) };
                        scope_stack.push(scope.to_string());
                    }
                } else if line.starts_with(b"$upscope") {
                    scope_stack.pop();
                } else if line.starts_with(b"$var") {
                    let tokens: Vec<&[u8]> = line
                        .split(|&b| b == b' ' || b == b'\t')
                        .filter(|s| !s.is_empty())
                        .collect();
                    if tokens.len() >= 5 {
                        let mut parts: Vec<&[u8]> = Vec::new();
                        for tok in tokens {
                            if tok == b"$end" {
                                break;
                            }
                            parts.push(tok);
                        }
                        if parts.len() >= 5 {
                            let var_type = unsafe { std::str::from_utf8_unchecked(parts[1]) }.to_string();
                            let size = unsafe { std::str::from_utf8_unchecked(parts[2]) }
                                .parse::<u32>()
                                .unwrap_or(0);
                            let id = parts[3];
                            let reference = unsafe { std::str::from_utf8_unchecked(parts[4]) }.to_string();
                            let mut range: Option<String> = None;
                            for tok in parts.iter().skip(5) {
                                if tok.starts_with(b"[") && tok.ends_with(b"]") {
                                    range = Some(unsafe { std::str::from_utf8_unchecked(tok) }.to_string());
                                    break;
                                }
                            }

                            let mut ref_with_range = reference.clone();
                            if let Some(r) = &range {
                                ref_with_range.push_str(r);
                            }

                            let scope = if scope_stack.is_empty() {
                                None
                            } else {
                                Some(scope_stack.join("."))
                            };

                            let full_name = if let Some(scope_name) = &scope {
                                format!("{}.{}", scope_name, ref_with_range)
                            } else {
                                ref_with_range.clone()
                            };

                            let signal_id = if let Some(&idx) = id_to_index.get(id) {
                                idx
                            } else {
                                let idx = id_to_index.len() as u32;
                                id_to_index.insert(id, idx);
                                let id_len = id.len();
                                if !id_lens.contains(&id_len) {
                                    id_lens.push(id_len);
                                }
                                idx
                            };

                            let id_code = unsafe { std::str::from_utf8_unchecked(id) }.to_string();
                            signal_meta.push(VcdSignalMeta {
                                signal_id,
                                full_name,
                                id_code,
                                var_type,
                                size,
                                range,
                                scope,
                                reference,
                            });
                        }
                    }
                }
            }
            if timescale_pending {
                if line.starts_with(b"$end") {
                    timescale_pending = false;
                } else {
                    timescale_raw = Some(unsafe { std::str::from_utf8_unchecked(line) }.trim().to_string());
                    timescale_seconds = timescale_raw
                        .as_deref()
                        .and_then(parse_timescale_seconds);
                    timescale_pending = false;
                }
            }
            cursor = end + 1;
        }

        id_lens.sort_unstable_by(|a, b| b.cmp(a));

        Ok(VcdHeader {
            body_start: cursor,
            id_to_index,
            id_lens_desc: id_lens,
            signal_meta,
            timescale_raw,
            timescale_seconds,
            date,
            version,
        })
    }

    fn parse_range_to_parquet(
        &self,
        header: &VcdHeader<'_>,
        parquet_path: &str,
        chunk_size: usize,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let data = &self.mmap[..];
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("signal_id", DataType::UInt32, false),
            Field::new("value_raw", DataType::Utf8, false),
            Field::new("value_kind", DataType::UInt8, false),
            Field::new("value_scalar", DataType::UInt8, true),
            Field::new("value_real", DataType::Float64, true),
        ]));
        self.parse_range_to_parquet_with_schema(
            data,
            &schema,
            &header.id_to_index,
            &header.id_lens_desc,
            parquet_path,
            chunk_size,
            start,
            end,
        )
    }

    fn parse_range_to_parquet_with_schema(
        &self,
        data: &[u8],
        schema: &Arc<Schema>,
        id_to_index: &FxHashMap<&[u8], u32>,
        id_lens_desc: &Vec<usize>,
        parquet_path: &str,
        chunk_size: usize,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
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

        let mut current_time: u64 = 0;
        let mut cursor = start;
        let end_limit = end.min(data.len());

        while cursor < end_limit {
            let line_end = memchr(b'\n', &data[cursor..end_limit])
                .map(|i| cursor + i)
                .unwrap_or(end_limit);
            let mut line = &data[cursor..line_end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }

            if !line.is_empty() {
                match line[0] {
                    b'#' => {
                        let time_str = unsafe { std::str::from_utf8_unchecked(&line[1..]) };
                        if let Ok(t) = time_str.parse::<u64>() {
                            current_time = t;
                        }
                    }
                    b'$' => {
                        // Skip VCD body directives ($dumpvars, $end, etc.)
                    }
                    b'b' | b'B' | b'r' | b'R' => {
                        if let Some(space_idx) = memchr(b' ', line) {
                            let value = &line[1..space_idx];
                            let id = &line[space_idx + 1..];
                            if let Some(&idx) = id_to_index.get(id) {
                                ts_builder.append_value(current_time);
                                id_builder.append_value(idx);
                                let raw = unsafe { std::str::from_utf8_unchecked(value) };
                                raw_builder.append_value(raw);
                                if line[0] == b'r' || line[0] == b'R' {
                                    kind_builder.append_value(2);
                                    scalar_builder.append_null();
                                    if let Ok(real_val) = raw.parse::<f64>() {
                                        real_builder.append_value(real_val);
                                    } else {
                                        real_builder.append_null();
                                    }
                                } else {
                                    kind_builder.append_value(1);
                                    scalar_builder.append_null();
                                    real_builder.append_null();
                                }
                            }
                        }
                    }
                    _ => {
                        // Scalar or strength value. Try fast path for 1-byte values.
                        if line.len() > 1 {
                            if let Some(&idx) = id_to_index.get(&line[1..]) {
                                ts_builder.append_value(current_time);
                                id_builder.append_value(idx);
                                let v = unsafe { std::str::from_utf8_unchecked(&line[0..1]) };
                                raw_builder.append_value(v);
                                kind_builder.append_value(0);
                                scalar_builder.append_value(map_scalar(v.as_bytes()[0]));
                                real_builder.append_null();
                            } else if let Some((idx, value)) = split_by_id_suffix(line, id_to_index, id_lens_desc) {
                                ts_builder.append_value(current_time);
                                id_builder.append_value(idx);
                                let raw = unsafe { std::str::from_utf8_unchecked(value) };
                                raw_builder.append_value(raw);
                                if raw.len() == 1 {
                                    kind_builder.append_value(0);
                                    scalar_builder.append_value(map_scalar(raw.as_bytes()[0]));
                                    real_builder.append_null();
                                } else {
                                    kind_builder.append_value(1);
                                    scalar_builder.append_null();
                                    real_builder.append_null();
                                }
                            }
                        }
                    }
                }
            }

            if ts_builder.len() >= chunk_size {
                flush_batch(
                    schema,
                    &mut writer,
                    &mut ts_builder,
                    &mut id_builder,
                    &mut raw_builder,
                    &mut kind_builder,
                    &mut scalar_builder,
                    &mut real_builder,
                )?;
            }

            cursor = line_end + 1;
        }

        if ts_builder.len() > 0 {
            flush_batch(
                schema,
                &mut writer,
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
}

fn write_vcd_meta_parquet(path: &str, signals: &[VcdSignalMeta]) -> PyResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("signal_id", DataType::UInt32, false),
        Field::new("signal_name", DataType::Utf8, false),
        Field::new("id_code", DataType::Utf8, false),
        Field::new("var_type", DataType::Utf8, false),
        Field::new("size", DataType::UInt32, false),
        Field::new("range", DataType::Utf8, true),
        Field::new("scope", DataType::Utf8, true),
        Field::new("reference", DataType::Utf8, false),
    ]));

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Meta Parquet init error: {}", e)))?;

    let mut signal_id_builder = UInt32Builder::with_capacity(signals.len());
    let mut name_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 16);
    let mut id_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 4);
    let mut var_type_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 8);
    let mut size_builder = UInt32Builder::with_capacity(signals.len());
    let mut range_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 8);
    let mut scope_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 16);
    let mut ref_builder = StringBuilder::with_capacity(signals.len(), signals.len() * 16);

    for meta in signals {
        signal_id_builder.append_value(meta.signal_id);
        name_builder.append_value(&meta.full_name);
        id_builder.append_value(&meta.id_code);
        var_type_builder.append_value(&meta.var_type);
        size_builder.append_value(meta.size);
        if let Some(range) = &meta.range {
            range_builder.append_value(range);
        } else {
            range_builder.append_null();
        }
        if let Some(scope) = &meta.scope {
            scope_builder.append_value(scope);
        } else {
            scope_builder.append_null();
        }
        ref_builder.append_value(&meta.reference);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(signal_id_builder.finish()),
            Arc::new(name_builder.finish()),
            Arc::new(id_builder.finish()),
            Arc::new(var_type_builder.finish()),
            Arc::new(size_builder.finish()),
            Arc::new(range_builder.finish()),
            Arc::new(scope_builder.finish()),
            Arc::new(ref_builder.finish()),
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

fn extract_inline_value(line: &[u8], keyword: &[u8]) -> Option<String> {
    let mut tokens = line
        .split(|&b| b == b' ' || b == b'\t')
        .filter(|s| !s.is_empty());
    let first = tokens.next()?;
    if first != keyword {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    for tok in tokens {
        if tok == b"$end" {
            break;
        }
        parts.push(unsafe { std::str::from_utf8_unchecked(tok) }.to_string());
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn parse_timescale_seconds(raw: &str) -> Option<f64> {
    let raw = raw.trim().replace(' ', "");
    if raw.is_empty() {
        return None;
    }
    let mut split_idx = 0;
    for (i, ch) in raw.chars().enumerate() {
        if ch.is_ascii_digit() {
            split_idx = i + 1;
        } else {
            break;
        }
    }
    if split_idx == 0 {
        return None;
    }
    let (num_str, unit_str) = raw.split_at(split_idx);
    let factor: f64 = num_str.parse().ok()?;
    let exp = match unit_str {
        "s" => 0,
        "ms" => -3,
        "us" => -6,
        "ns" => -9,
        "ps" => -12,
        "fs" => -15,
        _ => return None,
    };
    Some(factor * 10f64.powi(exp))
}

fn flush_batch(
    schema: &Arc<Schema>,
    writer: &mut ArrowWriter<File>,
    ts_builder: &mut UInt64Builder,
    id_builder: &mut UInt32Builder,
    raw_builder: &mut StringBuilder,
    kind_builder: &mut UInt8Builder,
    scalar_builder: &mut UInt8Builder,
    real_builder: &mut Float64Builder,
) -> PyResult<()> {
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ts_builder.finish()),
            Arc::new(id_builder.finish()),
            Arc::new(raw_builder.finish()),
            Arc::new(kind_builder.finish()),
            Arc::new(scalar_builder.finish()),
            Arc::new(real_builder.finish()),
        ],
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch error: {}", e)))?;
    writer
        .write(&batch)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Write error: {}", e)))?;
    Ok(())
}

fn split_by_id_suffix<'a>(
    line: &'a [u8],
    id_to_index: &FxHashMap<&'a [u8], u32>,
    id_lens_desc: &[usize],
) -> Option<(u32, &'a [u8])> {
    let line_len = line.len();
    for &len in id_lens_desc {
        if len >= line_len {
            continue;
        }
        let id_start = line_len - len;
        let id = &line[id_start..];
        if let Some(&idx) = id_to_index.get(id) {
            let value = &line[..id_start];
            if !value.is_empty() {
                return Some((idx, value));
            }
        }
    }
    None
}

fn find_next_timestamp_line(data: &[u8], mut pos: usize, end: usize) -> usize {
    let limit = end.min(data.len());
    if pos >= limit {
        return limit;
    }
    if pos > 0 && data[pos - 1] != b'\n' {
        if let Some(nl) = memchr(b'\n', &data[pos..limit]) {
            pos = pos + nl + 1;
        } else {
            return limit;
        }
    }

    let mut cursor = pos;
    while cursor < limit {
        let line_end = memchr(b'\n', &data[cursor..limit])
            .map(|i| cursor + i)
            .unwrap_or(limit);
        let mut line = &data[cursor..line_end];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        if !line.is_empty() && line[0] == b'#' {
            return cursor;
        }
        cursor = line_end + 1;
    }
    limit
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
