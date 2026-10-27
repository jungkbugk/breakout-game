"""돌파 매매 시뮬레이터 문제 생성.

기존 백테스트 데이터(backtest_v2/data)에서 돌파 사례를 찾아 문제 JSON으로 만든다.
- 시장 x 패턴 x 기간(2020 전/후)별로 '실전 비율 그대로' 무작위 추출 (성공/실패 균형 맞추지 않음)
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
WIN = 0.03      # 성공 기준(20일 후 수익률)
PER_GROUP = 50  # 시장 x 패턴 x 기간별 문제 수
MIN_GAP = 25    # 같은 종목 사례 간 최소 간격
SPLIT = pd.Timestamp("2020-01-01")

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
    """돌파 사례 (인덱스, 패턴, 거래량배수, 52주 고점 대비, 3개월 수익률) 목록. df: 정제된 일봉."""
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
        if not (c[i] > hi60[i] and v[i] >= 1.5 * vol20[i] and c[i] > o[i]):
            continue
        if val20[i] < min_value or i - last < MIN_GAP:
            continue
        if c[i] < ma50[i] * 0.98 or c[i] / c[i - 1] > 1.2:
            continue
        if c[i] > hi250[i]:
            pat = "A"
        elif r10[i] < 0.6 * r40[i] and c[i] > ma50[i]:
            pat = "C"
        elif box_hi[i] / box_lo[i] - 1 <= 0.28:
            pat = "B"
        elif c[i] <= 0.8 * hi250[i]:
            pat = "D"
        else:
            continue
        ev.append((i, pat, v[i] / vol20[i], c[i] / hi250[i] - 1, c[i] / c[i - 63] - 1))
        last = i
    return ev


def outcome(df, i):
    """돌파일 종가 진입, 손절 -8%, 20일 보유 후 청산."""
    c = df["Close"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    entry = c[i]
    stopped = False
    for k in range(1, HOLD + 1):
        if l[i + k] <= entry * (1 + STOP):
            stopped = True
            break
    ret = STOP if stopped else c[i + HOLD] / entry - 1
    path = c[i + 1:i + HOLD + 1] / entry - 1
    return ret, float(path.max()), float(path.min()), stopped


def pack(df, i, idx_frames):
    sl = df.iloc[i - PRE + 1:i + POST + 1]
    base = float(df["Close"].iloc[i])
    scale = 100.0 / base

    def r(a):
        return [round(float(x) * scale, 2) for x in a]

    vol = sl["Volume"].to_numpy(float)
    vmax = float(np.nanmax(vol[:PRE])) or 1.0
    q = {
        "o": r(sl["Open"]), "h": r(sl["High"]), "l": r(sl["Low"]), "c": r(sl["Close"]),
        "v": [int(round(x / vmax * 100)) for x in vol],
        "idx": {},
    }
    dates = sl.index
    for name, f in idx_frames.items():
        g = f["Close"].reindex(dates, method="ffill")
        ref = g.iloc[PRE - 1]
        q["idx"][name] = [round(float(x / ref * 100), 2) for x in g]
    return q, dates[PRE - 1]


def build_market(tag, frames, names, idx_frames, min_value, rng, state, rs, sp):
    frames = {k: clean(v) for k, v in frames.items()}
    groups, base = {}, {}
    for code, df in frames.items():
        for i, pat, vx, nh, r3 in find_events(df, min_value):
            ret, mx, mn, stopped = outcome(df, i)
            d = df.index[i]
            per = "pre" if d < SPLIT else "post"
            groups.setdefault((pat, per), []).append((code, df, i, ret, mx, mn, stopped, vx, nh, r3))
            b = base.setdefault(per, {}).setdefault(pat, [0, 0, 0.0])
            b[0] += 1
            b[1] += int(ret > WIN)
            b[2] += ret
    for per in base:
        for pat, b in base[per].items():
            b[2] = round(b[2] / b[0] * 100, 2)
    sg = state.reindex(rs.index, method="ffill")
    out = []
    for (pat, per), lst in sorted(groups.items()):
        rng.shuffle(lst)
        wins = sum(1 for x in lst if x[3] > WIN)
        print(tag, pat, per, "후보", len(lst), "성공률", round(wins / len(lst) * 100), "%")
        for code, df, i, ret, mx, mn, stopped, vx, nh, r3 in lst[:PER_GROUP]:
            q, date = pack(df, i, idx_frames)
            f = {"vx": round(vx, 1), "nh": round(nh * 100, 1), "r3": round(r3 * 100, 1)}
            if date in rs.index and code in rs.columns:
                v = rs.at[date, code]
                f["rs"] = int(round(v * 100)) if np.isfinite(v) else None
                if code in sp.columns:
                    v = sp.at[date, code]
                    f["sec"] = int(round(v * 100)) if np.isfinite(v) else None
                g = sg.loc[date]
                if np.isfinite(g["ok"]):
                    f["m"], f["d"] = int(g["ok"]), int(g["dist"])
            q.update({
                "mk": tag, "pat": pat, "per": per, "ok": bool(ret > WIN),
                "ret": round(ret * 100, 1), "max": round(mx * 100, 1), "min": round(mn * 100, 1),
                "stop": stopped, "code": code, "name": names.get(code, code),
                "date": date.strftime("%Y-%m-%d"), "f": f,
            })
            out.append(q)
    return out, base


def main():
    rng = random.Random(7)
    kr = load("kr_all.pkl")
    ks, kq = load("ks11.pkl"), load("kq11.pkl")
    wics = json.load(open(os.path.join(DATA, "kr_wics.json"), encoding="utf-8"))
    kr_frames = {k: clean(v) for k, v in kr["data"].items()}
    kr_sec = {c: s for c, s in wics.items() if s and s != "없음" and c in kr_frames}
    rs, sp = rank_panels(kr_frames, kr_sec)
    kr_out, kr_base = build_market("KR", kr_frames, kr["names"], {"코스피": ks, "코스닥": kq}, 2e9, rng,
                                   idx_state(ks["Close"], ks["Volume"]), rs, sp)

    us = load("us_ohlcv_1500.pkl")
    etf = load("etf.pkl")
    idx = {n: pd.DataFrame({"Close": etf[("Close", t)].dropna()}) for n, t in [("S&P500", "SPY"), ("나스닥", "QQQ")]}
    us_frames = {t: clean(pd.DataFrame({k: us[k][t] for k in ["Open", "High", "Low", "Close", "Volume"]}).dropna(subset=["Close"]))
                 for t in us["Close"].columns}
    us_sec = {t: s for t, s in load("sectors.pkl").items() if isinstance(s, str) and s and t in us_frames}
    rs, sp = rank_panels(us_frames, us_sec)
    us_out, us_base = build_market("US", us_frames, {}, idx, 1e7, rng,
                                   idx_state(etf[("Close", "SPY")].dropna(), etf[("Volume", "SPY")].dropna()), rs, sp)

    qs = kr_out + us_out
    rng.shuffle(qs)
    with open(os.path.join(HERE, "questions.json"), "w", encoding="utf-8") as f:
        json.dump({"patterns": PATTERNS, "pre": PRE, "post": POST, "hold": HOLD,
                   "base": {"KR": kr_base, "US": us_base}, "q": qs},
                  f, ensure_ascii=False, separators=(",", ":"))
    print("문제", len(qs), "KR", len(kr_out), "US", len(us_out),
          "크기MB", round(os.path.getsize(os.path.join(HERE, "questions.json")) / 1e6, 2))


if __name__ == "__main__":
    main()
