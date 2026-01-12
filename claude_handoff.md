# WaveToEarth Handoff (for Claude Code)

## 프로젝트 비전 (핵심)
목표는 **에이전틱 RTL 분석**이다.  
사람이 파형에서 “가설 검증에 필요한 신호/범위”를 반복적으로 수집·해석하는 과정을 LLM/Agent가 **툴링으로 자동화**하도록 만드는 것이 본질이다.

즉, 파서 고도화는 **중간 컴포넌트**이고, 최종 목표는:
- 에이전트가 “필요한 신호/범위”를 빠르게 추출
- raw 값 → semantic 요약/관계 분석
- 반복되는 디버깅 루프에서 “2) 신호 수집/해석”을 자동화

## 현재 파이프라인/구성
**Rust 고성능 파서 → Parquet(sharded) → DuckDB → FastAPI → CLI/Python API**

- `wavetoearth_core/`: Rust 기반 VCD/FST 파서 → Parquet 변환
- `server.py`: DuckDB 로딩/쿼리, 신호 resolve/fuzzy, inspect/analyze API
- `cli.py`: `wavetoearth` CLI (투명 모드)
- `wavetoearth.py`: Python API (`Client`, `Waveform`) + context manager
- `analysis/compare_iree_baremetal.py`: IREE vs baremetal 비교 분석 스크립트
- `tests/verify_vcdvcd.py`: vcdvcd와 결과 일치 검증

## CLI / Python API 요약
### CLI
```
wavetoearth /root/matmul_comparison/baremetal.fst \
  --from=750000 --to=800000 \
  --signals="gemmini.io_busy,core.csr.io_csr_stall" \
  --output text
```
- wildcard 지원: `*` `?`
- 짧은 이름 자동 resolve (suffix/fuzzy)
- cycle 기반 범위: `--clock`, `--start-cycle`, `--end-cycle`, `--edge`

### Python API
```python
from wavetoearth import Client, Waveform

with Client() as cli:
    w = cli.load("/root/matmul_comparison/baremetal.fst")
    res = w.inspect(["gemmini.io_busy"], ts_start=0, ts_end=1_000_000)

with Waveform("/root/matmul_comparison/baremetal.fst") as w:
    res = w.inspect(["core.csr.io_csr_stall"], ts_start=0, ts_end=1_000_000)
```

## 핵심 결과/검증
### 1) VCD vs vcdvcd 일치 검증
- `tests/verify_vcdvcd.py`로 **baremetal.vcd** 여러 구간 비교
- wavetoearth 결과 == vcdvcd 결과 **일치 확인**
- 검증 의존성: `requirements-verify.txt`

### 2) IREE vs Baremetal (실제 GEMM 구간)
분석 기준:  
`gemmini.ex_controller.io_busy` + `core.io_rocc_cmd_valid/ready` 핸드셰이크 창

**결론 요약**
- iree가 빠른 이유는 “연산량 감소”가 아니라 **idle gap 감소**  
- busy 총량은 거의 동일, **cmd_window가 iree에서 더 짧음**

**FST 기준 수치 (VCD와 동일 확인됨)**
- cmd_window 길이  
  - baremetal: `419,575,999`  
  - iree: `301,881,999`
- busy_in_cmd_window  
  - baremetal: `261,726,000`  
  - iree: `258,302,000`
- idle_in_cmd_window  
  - baremetal: `157,849,999`  
  - iree: `43,579,999`
- utilization (busy/cmd_window)  
  - baremetal: `0.624`  
  - iree: `0.856` (1.37x 개선)
- rocc_cmd count: `126 → 74`  
- gemmini_cmd count: `126 → 56`

요약: **iree는 더 적은 명령으로 Gemmini를 더 연속적으로 사용**한다.

### 3) FST/VCD 동일성 (IREE)
`WAVETOEART_CACHE_TAG=bench_v3`로 iree.fst vs iree.vcd 비교  
→ 주요 지표 완전 일치 확인.

## 데이터 위치
```
/root/matmul_comparison/
  baremetal.fst, baremetal.vcd
  iree.fst, iree.vcd
  *.parquet_shards_* (캐시)
```

## 유용한 환경변수
- `WAVETOEART_USE_SHARDS=1`
- `WAVETOEART_SHARDS=auto`
- `WAVETOEART_CACHE_TAG=bench_v3`
- `WAVETOEART_MAX_SHARDS`, `WAVETOEART_MIN_SHARDS`

## TODO / 주의 사항
- **FastVCD scope/real/strength 처리**: 사용자 요청사항. FST와 동등한 표현/메타데이터 보장 필요.
- **Cycle 기반 범위 검증 강화**: `--clock` 기준 타임스탬프→cycle 변환 정확성 확인.
- **Agentic UX**: raw+analysis JSON 출력 포맷 다듬기 (coactivity 등 관계 요약 강화).

---

# Resume Prompt (Claude Code용)
```
너는 /root/wavetoearth 레포에서 작업 중이다. 목표는 “에이전틱 RTL 분석”을 위한 툴링 완성이다.

현재는 Rust 파서 → Parquet(shards) → DuckDB → FastAPI → CLI/Python API까지 구축되어 있다.

해야 할 일:
1) FastVCD가 scope/real/strength 값을 제대로 처리하는지 확인하고, FST와 동일한 정보를 제공하도록 보강해라.
2) CLI/서버에서 cycle 기반 범위(--clock/--start-cycle/--end-cycle)의 정확성을 baremetal/iree 데이터로 검증해라.
3) Agentic UX 관점에서 inspect/analyze JSON 구조를 개선해라 (raw + semantic + 관계 분석).
4) 필요시 tests/verify_vcdvcd.py, analysis/compare_iree_baremetal.py를 사용해 결과를 검증해라.

데이터는 /root/matmul_comparison/에 있고, cached parquet는 *_parquet_shards_* 디렉토리에 있다.
```
