# main.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib


# =========================
# Utilidades generales
# =========================

SEED = 42
np.random.seed(SEED)

pd.set_option("display.max_columns", 200)


def ahora_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def asegurar_directorio(ruta: Path) -> None:
    ruta.mkdir(parents=True, exist_ok=True)


def convertir_a_json_safe(obj):
    """Convierte recursivamente tipos numpy/pandas a tipos serializables por json."""
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, _np.generic):  # np.bool_, np.float64, etc.
        return obj.item()

    if isinstance(obj, (_pd.Timestamp, _pd.Timedelta)):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): convertir_a_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [convertir_a_json_safe(v) for v in obj]

    return obj


# =========================
# Configuración
# =========================

@dataclass
class RutasProyecto:
    raiz: Path
    ruta_dataset_modelo: Path
    ruta_modelo: Path
    dir_outputs: Path
    dir_outputs_scoring: Path
    dir_outputs_opt: Path
    dir_outputs_monitoring: Path
    dir_modelos: Path


def resolver_rutas(raiz: Path) -> RutasProyecto:
    ruta_dataset_modelo = raiz / "data" / "processed" / "azure_pm" / "dataset_modelo.parquet"
    dir_modelos = raiz / "modelos"
    ruta_modelo = dir_modelos / "modelo_baseline_falla_30d.joblib"

    dir_outputs = raiz / "outputs"
    dir_outputs_scoring = dir_outputs / "scoring"
    dir_outputs_opt = dir_outputs / "optimizacion"
    dir_outputs_monitoring = dir_outputs / "monitoring"

    for d in [dir_modelos, dir_outputs, dir_outputs_scoring, dir_outputs_opt, dir_outputs_monitoring]:
        asegurar_directorio(d)

    return RutasProyecto(
        raiz=raiz,
        ruta_dataset_modelo=ruta_dataset_modelo,
        ruta_modelo=ruta_modelo,
        dir_outputs=dir_outputs,
        dir_outputs_scoring=dir_outputs_scoring,
        dir_outputs_opt=dir_outputs_opt,
        dir_outputs_monitoring=dir_outputs_monitoring,
        dir_modelos=dir_modelos,
    )


# =========================
# Carga / Validaciones
# =========================

def cargar_dataset_modelo(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe dataset_modelo.parquet en: {ruta}")
    df = pd.read_parquet(ruta)
    if "fecha" not in df.columns:
        raise ValueError("El dataset_modelo debe contener columna 'fecha' tipo datetime.")
    df["fecha"] = pd.to_datetime(df["fecha"])
    if "machineID" not in df.columns:
        raise ValueError("El dataset_modelo debe contener columna 'machineID'.")
    df = df.sort_values(["fecha", "machineID"]).reset_index(drop=True)
    return df


def cargar_artefacto_modelo(ruta: Path):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el modelo en: {ruta}\n"
            f"Corre: python main.py train (o ajusta la ruta/artefacto)."
        )
    artefacto = joblib.load(ruta)
    if not all(k in artefacto for k in ["modelo_calibrado", "columnas_features", "objetivo"]):
        raise ValueError("El artefacto del modelo debe contener: modelo_calibrado, columnas_features, objetivo.")
    return artefacto


# =========================
# Métricas drift (PSI/KS)
# =========================

def calcular_psi(serie_ref, serie_prd, n_bins: int = 10) -> float:
    a = pd.Series(serie_ref).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    b = pd.Series(serie_prd).replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    if len(a) < 50 or len(b) < 50:
        return np.nan

    cortes = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    if len(cortes) < 3:
        return np.nan

    a_counts, _ = np.histogram(a, bins=cortes)
    b_counts, _ = np.histogram(b, bins=cortes)

    a_perc = a_counts / max(a_counts.sum(), 1)
    b_perc = b_counts / max(b_counts.sum(), 1)

    eps = 1e-6
    a_perc = np.clip(a_perc, eps, None)
    b_perc = np.clip(b_perc, eps, None)

    return float(np.sum((b_perc - a_perc) * np.log(b_perc / a_perc)))


def calcular_ks(serie_ref, serie_prd) -> float:
    a = pd.Series(serie_ref).replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
    b = pd.Series(serie_prd).replace([np.inf, -np.inf], np.nan).dropna().astype(float).values

    if len(a) < 50 or len(b) < 50:
        return np.nan

    a = np.sort(a)
    b = np.sort(b)

    valores = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, valores, side="right") / len(a)
    cdf_b = np.searchsorted(b, valores, side="right") / len(b)

    return float(np.max(np.abs(cdf_a - cdf_b)))


# =========================
# TRAIN (baseline + calibración)
# =========================

def entrenar_modelo_lr_calibrado(
    dataset: pd.DataFrame,
    objetivo: str,
    columnas_features: Optional[List[str]] = None,
    porcentaje_train: float = 0.80,
    cv_calibracion: int = 3,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, average_precision_score

    if objetivo not in dataset.columns:
        raise ValueError(f"No existe objetivo '{objetivo}' en dataset_modelo.")

    # Si no se entregan features, inferimos: todas menos columnas obvias
    if columnas_features is None:
        excluir = {"fecha", "machineID", objetivo}
        columnas_features = [c for c in dataset.columns if c not in excluir]

    # Separación temporal
    dataset = dataset.sort_values(["fecha", "machineID"]).reset_index(drop=True)
    fechas = np.sort(dataset["fecha"].unique())
    if len(fechas) < 20:
        raise ValueError("Muy pocas fechas únicas para split temporal robusto.")

    idx_corte = int(len(fechas) * porcentaje_train)
    fecha_corte = fechas[idx_corte]

    train = dataset[dataset["fecha"] <= fecha_corte].copy()
    valid = dataset[dataset["fecha"] > fecha_corte].copy()

    X_train = train[columnas_features]
    y_train = train[objetivo]
    X_valid = valid[columnas_features]
    y_valid = valid[objetivo]

    # Tipos de columnas
    columnas_num = [c for c in columnas_features if pd.api.types.is_numeric_dtype(dataset[c])]
    columnas_cat = [c for c in columnas_features if c not in columnas_num]

    prep_num = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
    ])

    prep_cat = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocesamiento = ColumnTransformer(
        transformers=[
            ("num", prep_num, columnas_num),
            ("cat", prep_cat, columnas_cat),
        ],
        remainder="drop",
    )

    lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=SEED
    )

    pipeline_lr = Pipeline(steps=[
        ("prep", preprocesamiento),
        ("modelo", lr),
    ])

    # Entrenar base
    pipeline_lr.fit(X_train, y_train)

    # Calibración (isotonic) usando cv (compatibilidad sklearn)
    calibrador = CalibratedClassifierCV(
        estimator=pipeline_lr,
        method="isotonic",
        cv=cv_calibracion
    )
    calibrador.fit(X_valid, y_valid)

    # Métricas
    proba_valid = calibrador.predict_proba(X_valid)[:, 1]
    ap_valid = average_precision_score(y_valid, proba_valid)
    auc_valid = roc_auc_score(y_valid, proba_valid) if y_valid.nunique() >= 2 else np.nan

    metricas = {
        "filas_train": int(len(train)),
        "filas_valid": int(len(valid)),
        "ap_valid": round(float(ap_valid), 4),
        "auc_valid": round(float(auc_valid), 4) if auc_valid == auc_valid else None,
        "fecha_corte": str(pd.to_datetime(fecha_corte))[:10],
    }

    return calibrador, columnas_features, metricas


def comando_train(args, rutas: RutasProyecto):
    dataset = cargar_dataset_modelo(rutas.ruta_dataset_modelo)

    objetivo = args.objetivo
    if objetivo is None:
        # Default razonable
        objetivo = "falla_30d" if "falla_30d" in dataset.columns else None
    if objetivo is None:
        raise ValueError("No se pudo inferir el objetivo. Pasa --objetivo NOMBRE_COLUMNA.")

    modelo_calibrado, columnas_features, metricas = entrenar_modelo_lr_calibrado(
        dataset=dataset,
        objetivo=objetivo,
        columnas_features=None,
        porcentaje_train=args.porcentaje_train,
        cv_calibracion=args.cv_calibracion,
    )

    artefacto = {
        "modelo_calibrado": modelo_calibrado,
        "columnas_features": columnas_features,
        "objetivo": objetivo,
        "metricas_train_valid": metricas,
        "timestamp": ahora_str(),
    }

    joblib.dump(artefacto, rutas.ruta_modelo)
    print("✅ Modelo guardado en:", rutas.ruta_modelo.resolve())
    print("Métricas:", metricas)


# =========================
# SCORE
# =========================

def generar_scoring_operacional(
    dataset: pd.DataFrame,
    modelo,
    columnas_features: List[str],
    ultimos_dias: int = 14,
) -> pd.DataFrame:
    fecha_max = dataset["fecha"].max()
    fecha_min = fecha_max - pd.Timedelta(days=ultimos_dias - 1)

    ventana = dataset[(dataset["fecha"] >= fecha_min) & (dataset["fecha"] <= fecha_max)].copy()
    if len(ventana) == 0:
        raise ValueError("Ventana de scoring vacía. Ajusta ultimos_dias.")

    X = ventana[columnas_features]
    ventana["score"] = modelo.predict_proba(X)[:, 1]
    return ventana


def comando_score(args, rutas: RutasProyecto):
    dataset = cargar_dataset_modelo(rutas.ruta_dataset_modelo)
    artefacto = cargar_artefacto_modelo(rutas.ruta_modelo)

    modelo = artefacto["modelo_calibrado"]
    columnas_features = artefacto["columnas_features"]

    scoring = generar_scoring_operacional(
        dataset=dataset,
        modelo=modelo,
        columnas_features=columnas_features,
        ultimos_dias=args.ultimos_dias
    )

    ts = ahora_str()
    ruta_out = rutas.dir_outputs_scoring / f"scoring_{ts}.parquet"
    scoring.to_parquet(ruta_out, index=False)

    print("✅ Scoring guardado en:", ruta_out.resolve())
    print("Filas:", len(scoring), "| máquinas:", scoring["machineID"].nunique())
    print("Score mean:", round(float(scoring["score"].mean()), 4))


# =========================
# OPTIMIZE (greedy)
# =========================

def agregar_riesgo_por_maquina(scoring: pd.DataFrame) -> pd.DataFrame:
    riesgo = (
        scoring
        .groupby("machineID", as_index=False)
        .agg(
            fecha_ultima=("fecha", "max"),
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            score_p90=("score", lambda s: float(np.quantile(s, 0.90))),
        )
    )
    riesgo["prob_usada"] = riesgo["score_p90"]
    return riesgo.sort_values("prob_usada", ascending=False).reset_index(drop=True)


def calcular_costos(
    df_riesgo: pd.DataFrame,
    costo_falla_base: float,
    costo_mant_base: float,
    reduccion_riesgo: float,
    ruido: float = 0.20,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = df_riesgo.copy()

    df["factor"] = 1 + rng.normal(0, ruido, size=len(df))
    df["factor"] = df["factor"].clip(0.6, 1.6)

    df["costo_falla"] = (costo_falla_base * df["factor"]).round(2)
    df["costo_mant"] = (costo_mant_base * (0.9 + 0.2 * df["factor"])).round(2)

    df["costo_esperado_sin_mant"] = (df["prob_usada"] * df["costo_falla"]).round(2)
    df["prob_post"] = (df["prob_usada"] * (1 - reduccion_riesgo)).round(6)
    df["costo_esperado_con_mant"] = (df["costo_mant"] + df["prob_post"] * df["costo_falla"]).round(2)

    df["beneficio_esperado"] = (df["costo_esperado_sin_mant"] - df["costo_esperado_con_mant"]).round(2)
    return df


def optimizar_plan_greedy(
    df_costos: pd.DataFrame,
    capacidad: int,
    presupuesto: float,
) -> pd.DataFrame:
    df = df_costos.copy()
    df = df[df["beneficio_esperado"] > 0].reset_index(drop=True)
    if len(df) == 0:
        return df

    df["ratio"] = (df["beneficio_esperado"] / df["costo_mant"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.sort_values("ratio", ascending=False).reset_index(drop=True)

    seleccion = []
    costo_acum = 0.0
    n = 0

    for i, fila in df.iterrows():
        if n >= capacidad:
            break
        if costo_acum + float(fila["costo_mant"]) <= presupuesto:
            seleccion.append(i)
            costo_acum += float(fila["costo_mant"])
            n += 1

    return df.loc[seleccion].copy()


def comando_optimize(args, rutas: RutasProyecto):
    dataset = cargar_dataset_modelo(rutas.ruta_dataset_modelo)
    artefacto = cargar_artefacto_modelo(rutas.ruta_modelo)

    modelo = artefacto["modelo_calibrado"]
    columnas_features = artefacto["columnas_features"]

    scoring = generar_scoring_operacional(
        dataset=dataset,
        modelo=modelo,
        columnas_features=columnas_features,
        ultimos_dias=args.ultimos_dias
    )

    riesgo = agregar_riesgo_por_maquina(scoring)
    riesgo_costos = calcular_costos(
        df_riesgo=riesgo,
        costo_falla_base=args.costo_falla_base,
        costo_mant_base=args.costo_mant_base,
        reduccion_riesgo=args.reduccion_riesgo,
        ruido=args.ruido_costos
    )

    plan = optimizar_plan_greedy(
        df_costos=riesgo_costos,
        capacidad=args.capacidad,
        presupuesto=args.presupuesto
    )

    ts = ahora_str()
    ruta_plan = rutas.dir_outputs_opt / f"plan_mantencion_{ts}.csv"
    ruta_resumen = rutas.dir_outputs_opt / f"resumen_plan_{ts}.json"

    plan.to_csv(ruta_plan, index=False)

    resumen = {
        "timestamp": ts,
        "n_seleccionadas": int(len(plan)),
        "costo_mant_total": round(float(plan["costo_mant"].sum()), 2) if len(plan) else 0.0,
        "beneficio_total": round(float(plan["beneficio_esperado"].sum()), 2) if len(plan) else 0.0,
        "costo_sin_sel": round(float(plan["costo_esperado_sin_mant"].sum()), 2) if len(plan) else 0.0,
        "costo_con_sel": round(float(plan["costo_esperado_con_mant"].sum()), 2) if len(plan) else 0.0,
        "parametros": {
            "ultimos_dias": args.ultimos_dias,
            "capacidad": args.capacidad,
            "presupuesto": args.presupuesto,
            "costo_falla_base": args.costo_falla_base,
            "costo_mant_base": args.costo_mant_base,
            "reduccion_riesgo": args.reduccion_riesgo,
            "ruido_costos": args.ruido_costos,
        }
    }

    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_safe(resumen), f, indent=2, ensure_ascii=False)

    print("✅ Plan guardado en:", ruta_plan.resolve())
    print("✅ Resumen guardado en:", ruta_resumen.resolve())
    print("Resumen:", resumen)


# =========================
# MONITOR
# =========================

def performance_por_periodo(df: pd.DataFrame, objetivo: str, freq: str = "W") -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score, average_precision_score

    out = []
    df = df.copy()
    df["periodo"] = df["fecha"].dt.to_period(freq).astype(str)

    for periodo, g in df.groupby("periodo"):
        y = g[objetivo].values
        s = g["score"].values

        auc = np.nan
        if len(np.unique(y)) >= 2:
            auc = roc_auc_score(y, s)

        ap = average_precision_score(y, s)

        out.append({
            "periodo": periodo,
            "filas": int(len(g)),
            "rate_evento_pct": round(float(np.mean(y) * 100), 2),
            "auc": round(float(auc), 4) if auc == auc else np.nan,
            "ap": round(float(ap), 4),
            "score_mean": round(float(np.mean(s)), 4),
        })

    return pd.DataFrame(out).sort_values("periodo").reset_index(drop=True)


def comando_monitor(args, rutas: RutasProyecto):
    dataset = cargar_dataset_modelo(rutas.ruta_dataset_modelo)
    artefacto = cargar_artefacto_modelo(rutas.ruta_modelo)

    modelo = artefacto["modelo_calibrado"]
    columnas_features = artefacto["columnas_features"]
    objetivo = artefacto["objetivo"]

    # Ventanas
    dias_ref = args.dias_referencia
    dias_prd = args.dias_produccion

    fecha_max = dataset["fecha"].max()
    inicio_prd = fecha_max - pd.Timedelta(days=dias_prd - 1)
    fin_prd = fecha_max

    fin_ref = inicio_prd - pd.Timedelta(days=1)
    inicio_ref = fin_ref - pd.Timedelta(days=dias_ref - 1)

    ref = dataset[(dataset["fecha"] >= inicio_ref) & (dataset["fecha"] <= fin_ref)].copy()
    prd = dataset[(dataset["fecha"] >= inicio_prd) & (dataset["fecha"] <= fin_prd)].copy()

    if len(ref) == 0 or len(prd) == 0:
        raise ValueError("Ventanas ref/prd vacías. Ajusta dias_referencia/dias_produccion.")

    # Scores
    ref["score"] = modelo.predict_proba(ref[columnas_features])[:, 1]
    prd["score"] = modelo.predict_proba(prd[columnas_features])[:, 1]

    # Drift features
    columnas_num = [c for c in columnas_features if pd.api.types.is_numeric_dtype(dataset[c])]
    filas = []
    for col in columnas_num:
        psi = calcular_psi(ref[col], prd[col], n_bins=args.n_bins)
        ks = calcular_ks(ref[col], prd[col])
        filas.append({
            "feature": col,
            "psi": round(float(psi), 4) if psi == psi else np.nan,
            "ks": round(float(ks), 4) if ks == ks else np.nan,
            "media_ref": round(float(pd.to_numeric(ref[col], errors="coerce").mean()), 2),
            "media_prd": round(float(pd.to_numeric(prd[col], errors="coerce").mean()), 2),
        })

    drift_features = pd.DataFrame(filas).sort_values(["psi", "ks"], ascending=False).reset_index(drop=True)

    # Drift scores
    psi_score = calcular_psi(ref["score"], prd["score"], n_bins=args.n_bins)
    ks_score = calcular_ks(ref["score"], prd["score"])

    drift_scores = pd.DataFrame([{
        "psi_score": round(float(psi_score), 4) if psi_score == psi_score else np.nan,
        "ks_score": round(float(ks_score), 4) if ks_score == ks_score else np.nan,
        "score_mean_ref": round(float(ref["score"].mean()), 4),
        "score_mean_prd": round(float(prd["score"].mean()), 4),
        "score_std_ref": round(float(ref["score"].std()), 4),
        "score_std_prd": round(float(prd["score"].std()), 4),
    }])

    # Performance
    perf = performance_por_periodo(prd, objetivo=objetivo, freq=args.freq_performance)

    # Alertas base (reglas simples)
    UMBRAL_PSI_SCORE = args.umbral_psi_score
    UMBRAL_PSI_FEATURE = args.umbral_psi_feature
    UMBRAL_N_FEATURES = args.umbral_n_features
    UMBRAL_CAIDA_AP = args.umbral_caida_ap
    UMBRAL_CAMBIO_RATE = args.umbral_cambio_rate_pp

    from sklearn.metrics import average_precision_score, roc_auc_score

    ap_ref = average_precision_score(ref[objetivo], ref["score"])
    auc_ref = roc_auc_score(ref[objetivo], ref["score"]) if ref[objetivo].nunique() >= 2 else np.nan

    ap_prd = float(perf["ap"].dropna().iloc[-1]) if len(perf) else np.nan
    rate_ref = float(ref[objetivo].mean() * 100)
    rate_prd = float(prd[objetivo].mean() * 100)

    n_feat_drift = int((drift_features["psi"] >= UMBRAL_PSI_FEATURE).sum())

    alerta_score = (psi_score == psi_score) and (psi_score >= UMBRAL_PSI_SCORE)
    alerta_feat = n_feat_drift >= UMBRAL_N_FEATURES

    alerta_ap = False
    if ap_prd == ap_prd:
        alerta_ap = ap_prd <= (ap_ref - UMBRAL_CAIDA_AP)

    alerta_rate = abs(rate_prd - rate_ref) >= UMBRAL_CAMBIO_RATE

    alertas = {
        "psi_score": round(float(psi_score), 4) if psi_score == psi_score else None,
        "ks_score": round(float(ks_score), 4) if ks_score == ks_score else None,
        "ap_ref": round(float(ap_ref), 4),
        "auc_ref": round(float(auc_ref), 4) if auc_ref == auc_ref else None,
        "ap_prd_actual": round(float(ap_prd), 4) if ap_prd == ap_prd else None,
        "rate_ref_pct": round(rate_ref, 2),
        "rate_prd_pct": round(rate_prd, 2),
        "n_features_psi_ge_umbral": n_feat_drift,
        "alerta_score": bool(alerta_score),
        "alerta_feat": bool(alerta_feat),
        "alerta_ap": bool(alerta_ap),
        "alerta_rate": bool(alerta_rate),
    }

    total_alertas = sum([alerta_score, alerta_feat, alerta_ap, alerta_rate])
    if total_alertas >= 3:
        estado = "ROJO"
        accion = "Investigar drift + validar performance; considerar reentrenamiento inmediato."
    elif total_alertas == 2:
        estado = "AMARILLO"
        accion = "Monitoreo reforzado; revisar features críticas; preparar retraining si persiste."
    else:
        estado = "VERDE"
        accion = "Operación estable; continuar monitoreo normal."

    semaforo = {
        "estado": estado,
        "accion_recomendada": accion,
        "total_alertas": int(total_alertas),
        "detalle_alertas": {
            "score_drift": bool(alerta_score),
            "features_drift": bool(alerta_feat),
            "performance_decay": bool(alerta_ap),
            "label_drift": bool(alerta_rate),
        }
    }

    # Guardado
    ts = ahora_str()
    ruta_features = rutas.dir_outputs_monitoring / f"drift_features_{ts}.csv"
    ruta_scores = rutas.dir_outputs_monitoring / f"drift_scores_{ts}.csv"
    ruta_perf = rutas.dir_outputs_monitoring / f"performance_{ts}.csv"
    ruta_json = rutas.dir_outputs_monitoring / f"resumen_monitoreo_{ts}.json"

    drift_features.to_csv(ruta_features, index=False)
    drift_scores.to_csv(ruta_scores, index=False)
    perf.to_csv(ruta_perf, index=False)

    resumen = {
        "timestamp": ts,
        "ventana_referencia": {"inicio": str(inicio_ref)[:10], "fin": str(fin_ref)[:10], "filas": int(len(ref))},
        "ventana_produccion": {"inicio": str(inicio_prd)[:10], "fin": str(fin_prd)[:10], "filas": int(len(prd))},
        "alertas": alertas,
        "semaforo": semaforo,
        "top10_features_por_psi": drift_features[["feature", "psi", "ks", "media_ref", "media_prd"]].head(10).to_dict("records"),
    }

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(convertir_a_json_safe(resumen), f, indent=2, ensure_ascii=False)

    print("✅ Drift features:", ruta_features.resolve())
    print("✅ Drift scores  :", ruta_scores.resolve())
    print("✅ Performance   :", ruta_perf.resolve())
    print("✅ Resumen JSON  :", ruta_json.resolve())
    print("Semáforo:", semaforo)


# =========================
# CLI
# =========================

def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline Predictive Maintenance (train/score/optimize/monitor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    sub = parser.add_subparsers(dest="comando", required=True)

    # train
    p_train = sub.add_parser("train", help="Entrenar baseline LR + calibración y guardar modelo")
    p_train.add_argument("--objetivo", type=str, default=None, help="Nombre de columna objetivo (ej: falla_30d)")
    p_train.add_argument("--porcentaje_train", type=float, default=0.80, help="Corte temporal train vs valid")
    p_train.add_argument("--cv_calibracion", type=int, default=3, help="CV para calibración isotónica")

    # score
    p_score = sub.add_parser("score", help="Generar scoring operativo (últimos N días)")
    p_score.add_argument("--ultimos_dias", type=int, default=14, help="Ventana reciente para scoring")

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimizar plan de mantención (greedy) bajo restricciones")
    p_opt.add_argument("--ultimos_dias", type=int, default=14, help="Ventana de scoring reciente")
    p_opt.add_argument("--capacidad", type=int, default=12, help="Máximo de máquinas a intervenir")
    p_opt.add_argument("--presupuesto", type=float, default=350_000, help="Presupuesto total")
    p_opt.add_argument("--costo_falla_base", type=float, default=120_000, help="Costo base de falla")
    p_opt.add_argument("--costo_mant_base", type=float, default=25_000, help="Costo base de mantención")
    p_opt.add_argument("--reduccion_riesgo", type=float, default=0.65, help="Reducción de riesgo post-mantención")
    p_opt.add_argument("--ruido_costos", type=float, default=0.20, help="Ruido en costos por heterogeneidad")

    # monitor
    p_mon = sub.add_parser("monitor", help="Monitoreo drift/performance y semáforo")
    p_mon.add_argument("--dias_referencia", type=int, default=60, help="Ventana baseline")
    p_mon.add_argument("--dias_produccion", type=int, default=14, help="Ventana reciente")
    p_mon.add_argument("--n_bins", type=int, default=10, help="Bins para PSI")
    p_mon.add_argument("--freq_performance", type=str, default="W", help="Frecuencia performance (W/M)")
    p_mon.add_argument("--umbral_psi_score", type=float, default=0.25)
    p_mon.add_argument("--umbral_psi_feature", type=float, default=0.20)
    p_mon.add_argument("--umbral_n_features", type=int, default=8)
    p_mon.add_argument("--umbral_caida_ap", type=float, default=0.05)
    p_mon.add_argument("--umbral_cambio_rate_pp", type=float, default=10.0)

    return parser


def main():
    parser = construir_parser()
    args = parser.parse_args()

    rutas = resolver_rutas(Path.cwd())

    if args.comando == "train":
        comando_train(args, rutas)
    elif args.comando == "score":
        comando_score(args, rutas)
    elif args.comando == "optimize":
        comando_optimize(args, rutas)
    elif args.comando == "monitor":
        comando_monitor(args, rutas)
    else:
        raise ValueError(f"Comando no reconocido: {args.comando}")


if __name__ == "__main__":
    main()
