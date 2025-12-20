use wavetoearth_core::WaveParser; // Not exposed?
// Actually I can't access WaveParser if it's in lib.rs and cdylib.
// I have to copy the logic or make it pub.
// I will copy the logic for debugging.

use std::collections::HashMap;
use wellen::{GetItem, SignalValue, simple};

fn main() {
    let fst_path = "../tests/dummy.fst";
    eprintln!("Loading {}", fst_path);

    let waveform = match simple::read(fst_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to read: {}", e);
            return;
        }
    };

    // 1. Map hierarchy
    let mut id_to_name: HashMap<wellen::SignalRef, String> = HashMap::new();
    let hierarchy = waveform.hierarchy();

    fn walk_fst(
        hierarchy: &wellen::Hierarchy,
        scope_idx: wellen::ScopeRef,
        stack: &mut Vec<String>,
        map: &mut HashMap<wellen::SignalRef, String>
    ) {
        let scope = hierarchy.get(scope_idx);
        stack.push(scope.name(hierarchy).to_string());

        for var_ref in scope.vars(hierarchy) {
            let var = hierarchy.get(var_ref);
            let full_name = format!("{}.{}", stack.join("."), var.name(hierarchy));
            map.insert(var.signal_ref(), full_name);
        }

        for sub_scope in scope.scopes(hierarchy) {
            walk_fst(hierarchy, sub_scope, stack, map);
        }
        stack.pop();
    }

    let mut stack = Vec::new();
    for scope in hierarchy.scopes() {
        walk_fst(hierarchy, scope, &mut stack, &mut id_to_name);
    }

    eprintln!("Found {} signals", id_to_name.len());

    // 3. Dense Scan Strategy (Load chunks of signals)
    let all_signals: Vec<wellen::SignalRef> = id_to_name.keys().cloned().collect();
    let signal_batch_size = 1000;

    for (chunk_idx, signal_chunk) in all_signals.chunks(signal_batch_size).enumerate() {
        eprintln!("Processing chunk {}", chunk_idx);
        waveform.load_signals(signal_chunk);

        let mut last_offsets = vec![None; signal_chunk.len()];
        let time_table = waveform.time_table();
        eprintln!("Time table len: {}", time_table.len());

        for (t_idx, &time) in time_table.iter().enumerate() {
            let t_idx_u32 = t_idx as u32;

            for (sig_idx, &signal_ref) in signal_chunk.iter().enumerate() {
                 if let Some(signal) = waveform.get_signal(signal_ref) {
                     let offset_opt = signal.get_offset(t_idx_u32);

                     if offset_opt != last_offsets[sig_idx] {
                         if let Some(ref offset) = offset_opt {
                             let val_enum = signal.get_value_at(offset, 0);
                             let val_str = val_enum.to_bit_string().unwrap_or("?".to_string());
                             // Print first few updates
                             if t_idx < 5 {
                                 eprintln!("Change: Time={} Name={} Val={}", time, id_to_name.get(&signal_ref).unwrap(), val_str);
                             }
                         }
                         last_offsets[sig_idx] = offset_opt;
                     }
                 }
            }
        }
        waveform.unload_signals(signal_chunk);
    }
    eprintln!("Done.");
}
