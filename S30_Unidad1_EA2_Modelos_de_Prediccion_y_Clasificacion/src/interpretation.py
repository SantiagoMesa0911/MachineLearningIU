"""Interpretaciones en texto construidas a partir de los resultados."""

from typing import Dict, List

import pandas as pd

from .classification_model import TextLogisticClassifier
from .regression_model import LinearRegressionModel


def interpret_regression(model: LinearRegressionModel, metrics: Dict[str, float],
                         cv: Dict[str, float], folds: int, unit: str,
                         feature_units: Dict[str, str]) -> List[str]:
    target = model.target
    mae, rmse, r2 = metrics["mae"], metrics["rmse"], metrics["r2"]
    baseline, mean = metrics["baseline_mae"], metrics["target_mean"]
    ratio = rmse / mae if mae else float("nan")
    references = model.reference_categories()

    lines = [
        f"R² = {r2:.3f}: el modelo explica el {r2 * 100:.1f}% de la variabilidad "
        f"de {target} en los datos de prueba"
        + ("; el resto depende de factores que las variables no capturan."
           if r2 < 0.5 else "."),
        f"MAE = {mae:.2f} {unit}: en promedio la predicción se aleja {mae:.1f} "
        f"{unit} del valor real ({mae / mean * 100:.1f}% del valor medio, "
        f"{mean:.1f} {unit}).",
        f"RMSE = {rmse:.2f} {unit} ({ratio:.2f} veces el MAE): "
        + ("los errores son homogéneos." if ratio < 1.3 else
           "hay algunos errores grandes (comandos mucho más largos de lo "
           "esperado) que pesan más que el resto."),
        f"Referencia: predecir siempre la media daría MAE = {baseline:.2f} {unit}; "
        f"la regresión reduce el error en {(1 - mae / baseline) * 100:.1f}%.",
        f"Validación cruzada ({folds} pliegues sobre entrenamiento): R² = "
        f"{cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}"
        + ("; el desempeño es estable." if cv["r2_std"] < 0.05 else
           "; el desempeño varía bastante según la partición."),
    ]

    coefs = model.coefficients()
    n_num = len(model.numeric_features)
    for _, row in coefs.iloc[:n_num].iterrows():
        lines.append(
            f"{row['variable']}: {row['coef_por_unidad']:+.3f} {unit} de {target} "
            f"por cada {feature_units.get(row['variable'], 'unidad')} adicional "
            f"({row['coef_estandarizado']:+.2f} por "
            f"desviación estándar), con las demás variables constantes.")

    categorical = coefs.iloc[n_num:]
    if not categorical.empty:
        top = categorical.loc[categorical["coef_estandarizado"].abs().idxmax()]
        feature = next(f for f in model.categorical_features
                       if top["variable"].startswith(f + "_"))
        value = top["variable"][len(feature) + 1:]
        lines.append(
            f"Variable categórica con mayor efecto: {feature} = {value}, con "
            f"{top['coef_estandarizado']:+.2f} {unit} frente a la referencia "
            f"'{references[feature]}' con las mismas condiciones.")

    intercepts = model.intercepts()
    refs = ", ".join(f"{k}={v}" for k, v in references.items())
    lines.append(
        f"Intercepto = {intercepts['intercepto_estandarizado']:.2f} {unit}: valor "
        f"estimado para las categorías de referencia ({refs}) con variables "
        f"numéricas iguales a la media de entrenamiento. En unidades originales "
        f"({intercepts['intercepto_unidades_originales']:.2f}) corresponde a "
        f"variables en 0, sin sentido práctico.")
    return lines


def describe_class_balance(distribution: pd.Series) -> str:
    counts = ", ".join(f"{label}={count} ({count / distribution.sum():.0%})"
                       for label, count in distribution.items())
    if distribution.max() / distribution.min() <= 1.5:
        return (f"Distribución de clases: {counts}. Están balanceadas: la "
                f"exactitud es representativa.")
    return (f"Distribución de clases: {counts}. Están desbalanceadas: la "
            f"exactitud premia a la clase mayoritaria, así que se priorizan "
            f"recall y F1 macro (cada clase pesa igual) y se usa "
            f"class_weight='balanced'.")


def interpret_classification(model: TextLogisticClassifier,
                             metrics: Dict[str, object], cv: Dict[str, float],
                             folds: int) -> List[str]:
    matrix: pd.DataFrame = metrics["confusion_matrix"]
    report = metrics["report"]
    n_test = metrics["n_test"]
    correct = int(sum(matrix.iat[i, i] for i in range(len(matrix))))

    lines = [
        f"Exactitud = {metrics['accuracy']:.3f}: {correct} de {n_test} peticiones "
        f"de prueba se clasificaron correctamente.",
        f"F1 macro = {metrics['f1_macro']:.3f} (precisión macro = "
        f"{metrics['precision_macro']:.3f}, recall macro = "
        f"{metrics['recall_macro']:.3f}).",
    ]
    for label in model.classes:
        r = report[label]
        lines.append(f"Clase {label}: recall {r['recall']:.3f} (detecta "
                     f"{r['recall']:.0%} de los casos reales), precisión "
                     f"{r['precision']:.3f}, F1 {r['f1-score']:.3f}, "
                     f"{int(r['support'])} casos en prueba.")

    f1_by_class = {c: report[c]["f1-score"] for c in model.classes}
    lines.append(f"Mejor reconocida: {max(f1_by_class, key=f1_by_class.get)}; "
                 f"más difícil: {min(f1_by_class, key=f1_by_class.get)}.")

    errors = [(int(matrix.iat[i, j]), real, pred)
              for i, real in enumerate(model.classes)
              for j, pred in enumerate(model.classes) if i != j and matrix.iat[i, j]]
    if errors:
        errors.sort(reverse=True)
        lines.append("Errores (de más a menos frecuente): " + "; ".join(
            f"{n} {real} predicha(s) como {pred}" for n, real, pred in errors) + ".")
    else:
        lines.append("La matriz de confusión no muestra errores en prueba.")

    lines.append(f"Validación cruzada estratificada ({folds} pliegues sobre "
                 f"entrenamiento): F1 macro = {cv['f1_macro_mean']:.3f} ± "
                 f"{cv['f1_macro_std']:.3f}.")
    for label, features in model.top_features(5).items():
        lines.append(f"Términos que más empujan hacia {label}: {', '.join(features)}.")
    return lines
