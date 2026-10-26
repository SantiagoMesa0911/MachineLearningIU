"""Interpretaciones en texto construidas a partir de los resultados."""

from typing import Dict, List

import pandas as pd

from .classification_model import LogisticRegressionClassifier
from .regression_model import LinearRegressionModel


def interpret_regression(model: LinearRegressionModel, metrics: Dict[str, float],
                         cv: Dict[str, float], folds: int) -> List[str]:
    target = model.target
    mae, rmse, r2 = metrics["mae"], metrics["rmse"], metrics["r2"]
    baseline, mean = metrics["baseline_mae"], metrics["target_mean"]
    ratio = rmse / mae if mae else float("nan")
    reference = model.reference_category()

    lines = [
        f"R² = {r2:.3f}: el modelo explica el {r2 * 100:.1f}% de la variabilidad "
        f"de {target} en los datos de prueba.",
        f"MAE = {mae:.3f} cm: en promedio la predicción se aleja {mae:.2f} cm del "
        f"valor real ({mae / mean * 100:.1f}% del valor medio, {mean:.2f} cm).",
        f"RMSE = {rmse:.3f} cm ({ratio:.2f} veces el MAE): "
        + ("los errores son homogéneos, sin errores grandes aislados."
           if ratio < 1.3 else
           "algunos errores grandes pesan más que el resto."),
        f"Referencia: un modelo que siempre predice la media tendría MAE = "
        f"{baseline:.3f} cm; la regresión reduce el error en "
        f"{(1 - mae / baseline) * 100:.1f}%.",
        f"Validación cruzada ({folds} pliegues sobre entrenamiento): R² = "
        f"{cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}; "
        + ("el desempeño es estable entre particiones."
           if cv["r2_std"] < 0.05 else
           "el desempeño varía entre particiones."),
    ]

    coefs = model.coefficients()
    n_num = len(model.numeric_features)
    for i, row in coefs.iterrows():
        if i < n_num:
            lines.append(
                f"Coeficiente {row['variable']}: {row['coef_por_cm']:+.3f} cm de "
                f"{target} por cada cm adicional ({row['coef_estandarizado']:+.3f} "
                f"cm por desviación estándar), con las demás variables constantes.")
        else:
            category = row["variable"].split("_", 1)[1]
            lines.append(
                f"Coeficiente {row['variable']}: una flor {category} tiene en "
                f"promedio {row['coef_por_cm']:+.3f} cm de {target} frente a una "
                f"{reference} con las mismas medidas.")

    numeric = coefs.iloc[:n_num]
    strongest = numeric.loc[numeric["coef_estandarizado"].abs().idxmax(), "variable"]
    lines.append(f"Entre las medidas numéricas, la de mayor peso (|coef. "
                 f"estandarizado|) es {strongest}. Los coeficientes de especie "
                 f"(0/1) no son comparables en esa escala, pero muestran que la "
                 f"especie explica buena parte del largo del pétalo.")

    intercepts = model.intercepts()
    lines += [
        f"Intercepto (escala estandarizada) = "
        f"{intercepts['intercepto_estandarizado']:.3f} cm: {target} estimado "
        f"para una flor {reference} con medidas iguales a la media de "
        f"entrenamiento.",
        f"Intercepto en unidades originales = "
        f"{intercepts['intercepto_unidades_originales']:.3f} cm (flor {reference} "
        f"con todas las medidas en 0 cm): no tiene sentido físico, solo ubica "
        f"el plano de regresión.",
    ]
    return lines


def describe_class_balance(distribution: pd.Series) -> str:
    counts = ", ".join(f"{label}={count}" for label, count in distribution.items())
    balanced = distribution.max() / distribution.min() <= 1.5
    if balanced:
        return (f"Distribución de clases: {counts}. Las clases están "
                f"balanceadas, así que la exactitud es representativa; además "
                f"se reportan precisión, recall y F1 macro, que dan el mismo "
                f"peso a cada clase.")
    return (f"Distribución de clases: {counts}. Las clases están "
            f"desbalanceadas, así que la exactitud puede engañar; se priorizan "
            f"F1, recall y precisión macro.")


def interpret_classification(model: LogisticRegressionClassifier,
                             metrics: Dict[str, object], cv: Dict[str, float],
                             folds: int) -> List[str]:
    matrix: pd.DataFrame = metrics["confusion_matrix"]
    report = metrics["report"]
    n_test = metrics["n_test"]
    correct = int(sum(matrix.iat[i, i] for i in range(len(matrix))))

    lines = [
        f"Exactitud = {metrics['accuracy']:.3f}: {correct} de {n_test} flores de "
        f"prueba se clasificaron correctamente.",
        f"Precisión macro = {metrics['precision_macro']:.3f}, recall macro = "
        f"{metrics['recall_macro']:.3f}, F1 macro = {metrics['f1_macro']:.3f}.",
    ]

    f1_by_class = {c: report[c]["f1-score"] for c in model.classes}
    best = max(f1_by_class.values())
    worst = min(f1_by_class.values())
    best_classes = [c for c, v in f1_by_class.items() if v == best]
    lines.append(f"Clase(s) mejor reconocida(s): {', '.join(best_classes)} "
                 f"(F1 = {best:.3f}).")
    if worst < best:
        worst_classes = [c for c, v in f1_by_class.items() if v == worst]
        lines.append(f"Clase(s) con menor F1: {', '.join(worst_classes)} "
                     f"(F1 = {worst:.3f}).")

    errors = []
    for i, real in enumerate(model.classes):
        for j, predicted in enumerate(model.classes):
            count = int(matrix.iat[i, j])
            if i != j and count:
                errors.append(f"{count} flor(es) {real} clasificada(s) como {predicted}")
    if errors:
        lines.append("Errores de la matriz de confusión: " + "; ".join(errors) + ".")
    else:
        lines.append("La matriz de confusión no muestra errores en prueba.")

    lines.append(
        f"Validación cruzada estratificada ({folds} pliegues sobre "
        f"entrenamiento): exactitud = {cv['accuracy_mean']:.3f} ± "
        f"{cv['accuracy_std']:.3f}.")

    coefs = model.coefficients()
    for label, row in coefs.iterrows():
        feature = row.abs().idxmax()
        lines.append(f"Para la clase {label}, la variable más influyente es "
                     f"{feature} (coef. {row[feature]:+.3f} sobre datos "
                     f"estandarizados).")
    return lines
