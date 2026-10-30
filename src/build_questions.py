"""돌파 매매 시뮬레이터 문제 생성.

기존 백테스트 데이터(backtest_v2/data)에서 돌파 사례를 찾아 문제 JSON으로 만든다.
- 시장 x 패턴 x 기간(2020 전/후)별로 '실전 비율 그대로' 무작위 추출 (성공/실패 균형 맞추지 않음)
- 매수 시점은 피벗(직전 60일 고점) 돌파 순간: 체결가 = max(시가, 피벗), 갭 5% 초과 제외.
  돌파일 장 시작 전까지의 정보만으로 사례를 고른다(종가·거래량 조건 사용 안 함).
- 전체 사례의 실제 성공률·평균수익을 base로 함께 저장
- 정답 공개용 지표(거래량 배수, 52주 고점 대비, RS 순위, 업종 내 순위, 지수 상태) 포함
출력: breakout_game/questions.json (index.html에 합쳐지기 전 원본)
실행: python build_questions.py
"""
import json
import os
import pickle
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "backtest_v2", "data")

PRE = 130       # 돌파일 이전 표시 봉 수
POST = 40       # 결과 공개/매매 연습용 이후 봉 수
HOLD = 20       # 수익률 판정 보유일
STOP = -0.08    # 손절
STOPS = [5, 8, 12]   # 게임에서 고를 수 있는 손절폭(%) — 통계는 각각 따로 계산
WIN = 0.03      # 성공 기준(20일 후 수익률)
PER_GROUP = 50  # 시장 x 패턴 x 기간별 문제 수
MIN_GAP = 25    # 같은 종목 사례 간 최소 간격
SPLIT = pd.Timestamp("2020-01-01")
COST = {"KR": 0.004, "US": 0.003}   # 왕복 비용(수수료+세금+슬리피지) 비율

PATTERNS = {
    "A": "52주 신고가 돌파",
    "B": "박스권 돌파",
    "C": "변동성 수축 돌파",
    "D": "바닥 반등 돌파",
}


def load(path):
    with open(os.path.join(DATA, path), "rb") as f:
        return pickle.load(f)


def clean(df):
    """저가 0 등 데이터 결함 봉은 시가·고가·저가를 종가로 대체."""
    df = df[~df.index.duplicated()].dropna(subset=["Close"]).copy()
    o, h, l, c = (df[k].to_numpy(float) for k in ["Open", "High", "Low", "Close"])
    bad = (o <= 0) | (h <= 0) | (l <= 0) | (l > h) | (l < 0.5 * c) | (h > 1.6 * c)
    for k, a in (("Open", o), ("High", h), ("Low", l)):
        a = a.copy()
        a[bad] = c[bad]
        df[k] = a
    return df


def idx_state(close, vol):
    """지수 상태: 50일선 위·50일선 상승·200일선 상승 충족 개수, 최근 25일 분산일 수."""
    s50, s200 = close.rolling(50).mean(), close.rolling(200).mean()
    ok = (close > s50).astype(int) + (s50 > s50.shift(5)).astype(int) + (s200 > s200.shift(5)).astype(int)
    dist = ((close.pct_change() <= -0.002) & (vol > vol.shift(1))).astype(int).rolling(25).sum()
    return pd.DataFrame({"ok": ok, "dist": dist})


def rank_panels(frames, sec):
    """종합 RS 백분위(전체 종목) / 업종 내 백분위. 반환: (RS, SEC) DataFrame(날짜 x 종목)."""
    C = pd.DataFrame({k: v["Close"] for k, v in frames.items()}).sort_index().astype("float32")
    r = lambda n: C / C.shift(n) - 1
    comp = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
    rs = comp.rank(axis=1, pct=True)
    s = pd.Series(sec).reindex(C.columns).dropna()
    sp = comp[s.index].T.groupby(s).rank(pct=True).T
    return rs, sp


def find_events(df, min_value):
    """피벗 돌파 사례 (인덱스, 패턴, 거래량배수, 52주 고점 대비, 3개월 수익률, 피벗, 진입가). df: 정제된 일봉."""
    if len(df) < 320:
        return []
    o, h, l, c, v = (df[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    s = pd.Series
    hi60 = s(h).shift(1).rolling(60).max().to_numpy()
    hi250 = s(h).shift(1).rolling(250).max().to_numpy()
    vol20 = s(v).shift(1).rolling(20).mean().to_numpy()
    val20 = s(c * v).shift(1).rolling(20).mean().to_numpy()
    ma50 = s(c).rolling(50).mean().to_numpy()
    box_hi = s(h).shift(1).rolling(40).max().to_numpy()
    box_lo = s(l).shift(1).rolling(40).min().to_numpy()
    rng = (h - l) / np.where(c > 0, c, np.nan)
    r10 = s(rng).shift(1).rolling(10).mean().to_numpy()
    r40 = s(rng).shift(11).rolling(30).mean().to_numpy()

    ev, last = [], -10**9
    n = len(df)
    for i in range(260, n - POST):
        P = hi60[i]
        cp = c[i - 1]
        # 전일까지의 정보: 피벗 근접(-10% 이내), 50일선 부근 이상. 당일은 피벗 도달 여부와 갭만 사용.
        if not (h[i] >= P and o[i] <= 1.05 * P and 0.9 * P <= cp < P and cp >= ma50[i - 1] * 0.98):
            continue
        if val20[i] < min_value or i - last < MIN_GAP:
            continue
        if P >= 0.999 * hi250[i]:
            pat = "A"
        elif r10[i] < 0.6 * r40[i] and cp > ma50[i - 1]:
            pat = "C"
        elif box_hi[i] / box_lo[i] - 1 <= 0.28:
            pat = "B"
        elif cp <= 0.8 * hi250[i]:
            pat = "D"
        else:
            continue
        E = max(o[i], P)
        ev.append((i, pat, v[i] / vol20[i], cp / hi250[i] - 1, cp / c[i - 64] - 1, P, E))
        last = i
    return ev


def outcome(df, i, E, P, st=STOP):
    """피벗 체결가 E 진입, 손절 -8%(갭하락 시 시가 청산), 진입일 포함 20거래일 보유 후 종가 청산.
    진입일: 시가<피벗인 상승일은 저가가 체결 전에 나온 것으로 보아 손절을 적용하지 않는다."""
    o, l, c = (df[k].to_numpy(float) for k in ["Open", "Low", "Close"])
    stop = E * (1 + st)
    ret, stopped = None, False
    for k in range(0, HOLD + 1):
        j = i + k
        if k > 0 and o[j] <= stop:
            ret, stopped = o[j] / E - 1, True
            break
        if l[j] <= stop and (k > 0 or o[j] >= P or c[j] < o[j]):
            ret, stopped = st, True
            break
    if ret is None:
        ret = c[i + HOLD] / E - 1
    path = c[i:i + HOLD + 1] / E - 1
    return ret, float(path.max()), float(path.min()), stopped


def pack(df, i, E, P, idx_frames):
    """앞 PRE개 봉은 돌파 전날까지, 이후 POST개 봉은 진입일(i)부터. 가격은 진입가=100 기준."""
    sl = df.iloc[i - PRE:i + POST]
    scale = 100.0 / E

    def r(a):
        return [round(float(x) * scale, 2) for x in a]

    vol = sl["Volume"].to_numpy(float)
    vmax = float(np.nanmax(vol[:PRE])) or 1.0
    q = {
        "o": r(sl["Open"]), "h": r(sl["High"]), "l": r(sl["Low"]), "c": r(sl["Close"]),
        "v": [int(round(x / vmax * 100)) for x in vol],
        "idx": {}, "pv": round(P * scale, 2),
    }
    dates = sl.index
    for name, f in idx_frames.items():
        g = f["Close"].reindex(dates, method="ffill")
        ref = g.iloc[PRE - 1]
        q["idx"][name] = [round(float(x / ref * 100), 2) for x in g]
    return q, df.index[i]


# 지표 구간 (키, 이름, [(라벨, 하한, 상한)]) — 하한 이상 상한 미만이면 해당 구간
INF = float("inf")
FEATS = [
    ("vx", "돌파일 거래량(20일 평균 대비)", [("1배 미만", -INF, 1), ("1~1.5배", 1, 1.5), ("1.5~2배", 1.5, 2), ("2~3배", 2, 3), ("3배 이상", 3, INF)]),
    ("nh", "52주 고점 대비", [("고점 -3% 이내", -3, INF), ("-3~-10%", -10, -3), ("-10~-20%", -20, -10), ("-20% 이하", -INF, -20)]),
    ("r3", "최근 3개월 수익률", [("0% 미만", -INF, 0), ("0~20%", 0, 20), ("20~50%", 20, 50), ("50% 이상", 50, INF)]),
    ("rs", "종합 RS(전 종목)", [("상위 10% 이내", 90, INF), ("상위 10~20%", 80, 90), ("상위 20~50%", 50, 80), ("하위 50%", -INF, 50)]),
    ("sec", "업종 내 위치", [("상위 10% 이내", 90, INF), ("상위 10~20%", 80, 90), ("상위 20~50%", 50, 80), ("하위 50%", -INF, 50)]),
    ("m", "지수 상태(충족 개수)", [("0~1 / 3", -INF, 2), ("2 / 3", 2, 3), ("3 / 3", 3, INF)]),
    ("d", "지수 분산일(25일)", [("0~2개", -INF, 3), ("3~5개", 3, 6), ("6개 이상", 6, INF)]),
]
RULES = ["h20", "h40", "ma5", "ma10", "ma20", "ema50", "pyr"]


def bucket_of(key, v):
    if v is None:
        return -1
    for k, _, bs in FEATS:
        if k == key:
            for bi, (_, lo, hi) in enumerate(bs):
                if lo <= v < hi:
                    return bi
    return -1


def ma_series(c, kind, n):
    a = np.asarray(c, float)
    if kind == "ema":
        k = 2 / (n + 1)
        out = np.empty(len(a))
        p = 0.0
        for i, x in enumerate(a):
            p = x if i == 0 else x * k + p * (1 - k)
            out[i] = p
        return out
    cs = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full(len(a), np.nan)
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def sim_rule(o, h, l, c, pv, rule, S=8):
    """진입가=100 기준 배열로 청산 규칙을 시뮬레이션 -> (손익%, 청산 봉 인덱스, 투입 단위 수).
    template.html의 simRule과 동일해야 한다. 손절 -8%는 모든 규칙에 포함."""
    e, L = PRE, len(c)
    lv = 100 - S
    estop = o[e] >= pv - 1e-6 or c[e] < o[e]
    ma = None
    if rule.startswith("ma"):
        ma = ma_series(c, "sma", int(rule[2:]))
    elif rule == "ema50":
        ma = ma_series(c, "ema", 50)
    elif rule == "pyr":
        ma = ma_series(c, "sma", 10)
    add = None

    def pn(px):
        return px - 100 + ((px - add) if add is not None else 0.0)

    for j in range(e, L):
        un = 2 if add is not None else 1
        if j > e and o[j] <= lv:
            return pn(o[j]), j, un
        if l[j] <= lv and (j > e or estop):
            return pn(float(lv)), j, un
        if rule == "h20" and j == e + 20:
            return pn(c[j]), j, un
        if rule == "pyr" and add is None and c[j] >= 105:
            add = c[j]
            un = 2
        if ma is not None and not np.isnan(ma[j]) and c[j] < ma[j]:
            if j + 1 < L:
                return pn(o[j + 1]), j + 1, un
            return pn(c[j]), j, un
    return pn(c[L - 1]), L - 1, (2 if add is not None else 1)


def scaled_arrays(df, i, E):
    sl = df.iloc[i - PRE:i + POST]
    s = 100.0 / E
    return [np.round(sl[k].to_numpy(float) * s, 2) for k in ("Open", "High", "Low", "Close")]


def feats_of(prev, code, vx, nh, r3, rs, sp, sg):
    f = {"vx": round(vx, 1), "vok": bool(vx >= 1.5), "nh": round(nh * 100, 1), "r3": round(r3 * 100, 1)}
    if prev in rs.index and code in rs.columns:
        v = rs.at[prev, code]
        f["rs"] = int(round(v * 100)) if np.isfinite(v) else None
        if code in sp.columns:
            v = sp.at[prev, code]
            f["sec"] = int(round(v * 100)) if np.isfinite(v) else None
        g = sg.loc[prev]
        if np.isfinite(g["ok"]):
            f["m"], f["d"] = int(g["ok"]), int(g["dist"])
    f["b"] = [bucket_of(k, f.get(k)) for k, _, _ in FEATS]
    return f


def finalize(base, fs, rl, cost):
    """누적 통계를 JSON용으로 정리."""
    for per in base:
        for pat, b in base[per].items():
            n = b[0]
            b[2] = round(b[2] / n * 100, 2)
            b[4] = round(b[4] / n * 100, 2)
    # 지표 구간 통계: [라벨, 건수, 성공률%, 평균%, 비용반영 성공률%, 비용반영 평균%]
    fstat = {}
    for k, nm, bs in FEATS:
        rows = []
        for (lab, _, _), s in zip(bs, fs[k]):
            n = s[0]
            rows.append([lab, n, round(s[1] / n * 100, 1) if n else None, round(s[2] / n * 100, 2) if n else None,
                         round(s[3] / n * 100, 1) if n else None, round(s[4] / n * 100, 2) if n else None])
        fstat[k] = {"name": nm, "rows": rows}
    # 청산 규칙 통계: [건수, 평균, 이익비율, 평균이익, 평균손실, 평균보유일, 비용반영 이익비율, 비용반영 평균, 평균 투입단위]
    rstat = {}
    for r, lst in rl.items():
        a = np.array(lst, float)
        pnl, un, hold = a[:, 0], a[:, 1], a[:, 2]
        net = pnl - cost * un
        w, ls = pnl[pnl > 0], pnl[pnl <= 0]
        rstat[r] = [len(a), round(float(pnl.mean()), 2), round(float((pnl > 0).mean() * 100), 1),
                    round(float(w.mean()), 2) if len(w) else 0, round(float(ls.mean()), 2) if len(ls) else 0,
                    round(float(hold.mean()), 1), round(float((net > 0).mean() * 100), 1), round(float(net.mean()), 2),
                    round(float(un.mean()), 2)]
    return base, fstat, rstat


def build_market(tag, frames, names, idx_frames, min_value, rng, state, rs, sp):
    frames = {k: clean(v) for k, v in frames.items()}
    sg = state.reindex(rs.index, method="ffill")
    cost = COST[tag] * 100
    groups = {}
    base = {S: {} for S in STOPS}
    fs = {S: {k: [[0, 0, 0.0, 0, 0.0] for _ in bs] for k, _, bs in FEATS} for S in STOPS}
    rl = {S: {r: [] for r in RULES} for S in STOPS}
    for code, df in frames.items():
        for i, pat, vx, nh, r3, P, E in find_events(df, min_value):
            d = df.index[i]
            per = "pre" if d < SPLIT else "post"
            f = feats_of(df.index[i - 1], code, vx, nh, r3, rs, sp, sg)
            o_, h_, l_, c_ = scaled_arrays(df, i, E)
            pv = P * 100.0 / E
            for S in STOPS:
                ret, mx, mn, stopped = outcome(df, i, E, P, -S / 100)
                if S == 8:
                    groups.setdefault((pat, per), []).append((code, df, i, ret, mx, mn, stopped, f, P, E))
                b = base[S].setdefault(per, {}).setdefault(pat, [0, 0, 0.0, 0, 0.0])
                b[0] += 1
                b[1] += int(ret > WIN)
                b[2] += ret
                b[3] += int(ret - COST[tag] > WIN)
                b[4] += ret - COST[tag]
                for (k, _, _), bi in zip(FEATS, f["b"]):
                    if bi >= 0:
                        s = fs[S][k][bi]
                        s[0] += 1
                        s[1] += int(ret > WIN)
                        s[2] += ret
                        s[3] += int(ret - COST[tag] > WIN)
                        s[4] += ret - COST[tag]
                for r in RULES:
                    pnl, jj, un = sim_rule(o_, h_, l_, c_, pv, r, S)
                    rl[S][r].append((pnl, un, jj - PRE))
    stats = {S: finalize(base[S], fs[S], rl[S], cost) for S in STOPS}
    out = []
    for (pat, per), lst in sorted(groups.items()):
        rng.shuffle(lst)
        wins = sum(1 for x in lst if x[3] > WIN)
        print(tag, pat, per, "후보", len(lst), "성공률", round(wins / len(lst) * 100), "%")
        for code, df, i, ret, mx, mn, stopped, f, P, E in lst[:PER_GROUP]:
            q, date = pack(df, i, E, P, idx_frames)
            q.update({
                "mk": tag, "pat": pat, "per": per, "ok": bool(ret > WIN),
                "ret": round(ret * 100, 1), "max": round(mx * 100, 1), "min": round(mn * 100, 1),
                "stop": stopped, "code": code, "name": names.get(code, code),
                "date": date.strftime("%Y-%m-%d"), "f": f,
            })
            out.append(q)
    return out, stats


def main():
    rng = random.Random(7)
    kr = load("kr_all.pkl")
    ks, kq = load("ks11.pkl"), load("kq11.pkl")
    wics = json.load(open(os.path.join(DATA, "kr_wics.json"), encoding="utf-8"))
    kr_frames = {k: clean(v) for k, v in kr["data"].items()}
    kr_sec = {c: s for c, s in wics.items() if s and s != "없음" and c in kr_frames}
    rs, sp = rank_panels(kr_frames, kr_sec)
    kr_out, kr_st = build_market("KR", kr_frames, kr["names"], {"코스피": ks, "코스닥": kq}, 2e9, rng,
                                   idx_state(ks["Close"], ks["Volume"]), rs, sp)

    us = load("us_ohlcv_1500.pkl")
    etf = load("etf.pkl")
    idx = {n: pd.DataFrame({"Close": etf[("Close", t)].dropna()}) for n, t in [("S&P500", "SPY"), ("나스닥", "QQQ")]}
    us_frames = {t: clean(pd.DataFrame({k: us[k][t] for k in ["Open", "High", "Low", "Close", "Volume"]}).dropna(subset=["Close"]))
                 for t in us["Close"].columns}
    us_sec = {t: s for t, s in load("sectors.pkl").items() if isinstance(s, str) and s and t in us_frames}
    rs, sp = rank_panels(us_frames, us_sec)
    us_out, us_st = build_market("US", us_frames, {}, idx, 1e7, rng,
                                   idx_state(etf[("Close", "SPY")].dropna(), etf[("Volume", "SPY")].dropna()), rs, sp)

    qs = kr_out + us_out
    rng.shuffle(qs)
    with open(os.path.join(HERE, "questions.json"), "w", encoding="utf-8") as f:
        json.dump({"patterns": PATTERNS, "pre": PRE, "post": POST, "hold": HOLD,
                   "base": {"KR": kr_st[8][0], "US": us_st[8][0]}, "cost": COST,
                   "fs": {"KR": kr_st[8][1], "US": us_st[8][1]}, "rules": {"KR": kr_st[8][2], "US": us_st[8][2]},
                   "alt": {str(S): {"base": {"KR": kr_st[S][0], "US": us_st[S][0]},
                                    "fs": {"KR": kr_st[S][1], "US": us_st[S][1]},
                                    "rules": {"KR": kr_st[S][2], "US": us_st[S][2]}} for S in STOPS if S != 8},
                   "featKeys": [k for k, _, _ in FEATS], "q": qs},
                  f, ensure_ascii=False, separators=(",", ":"))
    print("문제", len(qs), "KR", len(kr_out), "US", len(us_out),
          "크기MB", round(os.path.getsize(os.path.join(HERE, "questions.json")) / 1e6, 2))


if __name__ == "__main__":
    main()
