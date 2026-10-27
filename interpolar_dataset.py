"""
=============================================================================
MÓDULO DE INTERPOLACIÓN TEMPORAL DE LECTURAS DS18B20 (NANO-IA)
=============================================================================
Objetivo:
    Homogeneizar la frecuencia de muestreo de 'dataset_temperatura_crudo_1.csv'
    (originalmente a ~10 segundos) para llevarlo a una resolución de ~5 segundos,
    igualando así la frecuencia de 'dataset_temperatura_crudo_2.csv'.

Funcionamiento:
    - Detecta lecturas consecutivas separadas por un intervalo nominal de ~10s.
    - Genera un punto intermedio exacto a +5 segundos entre cada par.
    - Interpola linealmente la temperatura ('temp_c'), la hora decimal y el epoch.
    - Genera la trama serial sintética ('TEMP:XX.XXXX') compatible con data_cleaner.py.
    - Respeta pausas prolongadas (apagado de sensor / sesiones distintas) y
      ráfagas instantáneas (0 segundos) para no falsear los datos físicos.
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd


def interpolar_temperatura_5s(
    archivo_entrada: str = "dataset_temperatura_crudo_1.csv",
    archivo_salida: str = "dataset_temperatura_crudo_1_interpolado.csv",
    min_gap_segundos: float = 6.0,
    max_gap_segundos: float = 16.0,
    interpolar_todo: bool = False,
) -> pd.DataFrame:
    """
    Carga el CSV crudo a 10s e interpola un punto intermedio (a ~5s) entre cada lectura.

    Parámetros:
        archivo_entrada (str): Ruta del archivo CSV original de 10s.
        archivo_salida (str): Ruta donde se guardará el nuevo CSV interpolado.
        min_gap_segundos (float): Límite inferior de delta temporal para interpolar (ignora ráfagas de 0s).
        max_gap_segundos (float): Límite superior de delta temporal para interpolar (ignora pausas de sesión).
        interpolar_todo (bool): Si es True, fuerza la interpolación entre todo par adyacente.

    Retorna:
        pd.DataFrame: DataFrame con las lecturas originales e interpoladas combinadas y ordenadas.
    """
    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"No se encontró el archivo de entrada: '{archivo_entrada}'")

    print(f"[*] Leyendo archivo original: {archivo_entrada}")
    df = pd.read_csv(archivo_entrada)

    # Validar columnas requeridas
    columnas_esperadas = [
        "timestamp_iso", "epoch_segundos", "fecha", "hora_decimal",
        "lectura_serial_cruda", "temp_c", "es_perturbacion_manual"
    ]
    for col in columnas_esperadas:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida '{col}' en el CSV.")

    n_original = len(df)
    print(f"[*] Total de filas originales: {n_original}")

    # Calcular la diferencia temporal en segundos entre lecturas consecutivas
    diff_epoch = df["epoch_segundos"].shift(-1) - df["epoch_segundos"]

    # Máscara para decidir dónde se debe intercalar un punto intermedio
    if interpolar_todo:
        # Interpolar entre cualquier par válido
        mascara_interpolar = diff_epoch.notna() & (diff_epoch > 0)
    else:
        # Interpolar únicamente donde la diferencia es la de muestreo normal (~10s)
        # Esto previene inventar datos irreales en pausas de horas/días o ráfagas de 0s
        mascara_interpolar = diff_epoch.between(min_gap_segundos, max_gap_segundos)

    indices_base = df.index[mascara_interpolar]
    total_interpolados = len(indices_base)
    print(f"[*] Pares de lecturas a ~10s identificados: {total_interpolados}")
    print(f"[*] Puntos nuevos a ~5s que se generarán : {total_interpolados}")

    if total_interpolados == 0:
        print("[!] No se encontraron lecturas que cumplan el criterio de interpolación.")
        return df

    # =========================================================================
    # CÁLCULO VECTORIZADO DE LOS PUNTOS INTERMEDIOS CON PANDAS Y NUMPY
    # =========================================================================
    
    # 1. Epoch segundos (punto medio exacto)
    epoch_i = df.loc[indices_base, "epoch_segundos"].values
    epoch_next = df.loc[indices_base + 1, "epoch_segundos"].values
    mid_epoch = np.round((epoch_i + epoch_next) / 2.0, 2)

    # 2. Temperatura (interpolación lineal redondeada a 4 decimales, resolución DS18B20)
    temp_i = df.loc[indices_base, "temp_c"].values
    temp_next = df.loc[indices_base + 1, "temp_c"].values
    mid_temp = np.round((temp_i + temp_next) / 2.0, 4)

    # 3. Timestamps Datetime e ISO (redondeado al segundo más cercano)
    ts_i = pd.to_datetime(df.loc[indices_base, "timestamp_iso"])
    ts_next = pd.to_datetime(df.loc[indices_base + 1, "timestamp_iso"])
    mid_ts = pd.to_datetime(ts_i.values + (ts_next.values - ts_i.values) / 2).round("s")

    mid_iso = mid_ts.strftime("%Y-%m-%dT%H:%M:%S")
    mid_fecha = mid_ts.strftime("%Y-%m-%d")

    # 4. Hora decimal: calculada exactamente como H + M/60 + S/3600 con 4 decimales
    mid_hora = np.round(
        mid_ts.hour + mid_ts.minute / 60.0 + mid_ts.second / 3600.0, 4
    )

    # 5. Lectura serial cruda: con formato idéntico de Arduino "TEMP:XX.XXXX"
    mid_serial = [f"TEMP:{t:.4f}" for t in mid_temp]

    # 6. Perturbación manual: 1 si alguna de las lecturas contiguas era perturbación, 0 si no
    pert_i = df.loc[indices_base, "es_perturbacion_manual"].values
    pert_next = df.loc[indices_base + 1, "es_perturbacion_manual"].values
    mid_perturbacion = np.maximum(pert_i, pert_next)

    # Crear el DataFrame de puntos interpolados
    # Asignamos índice i + 0.5 para que al ordenar por índice quede exactamente en medio
    df_intermedio = pd.DataFrame(
        {
            "timestamp_iso": mid_iso,
            "epoch_segundos": mid_epoch,
            "fecha": mid_fecha,
            "hora_decimal": mid_hora,
            "lectura_serial_cruda": mid_serial,
            "temp_c": mid_temp,
            "es_perturbacion_manual": mid_perturbacion,
        },
        index=indices_base + 0.5,
    )

    # =========================================================================
    # FUSIÓN Y ORDENAMIENTO
    # =========================================================================
    df_combinado = pd.concat([df, df_intermedio])
    df_combinado = df_combinado.sort_index().reset_index(drop=True)

    # Guardar en CSV asegurando el mismo orden de columnas
    df_combinado = df_combinado[columnas_esperadas]
    df_combinado.to_csv(archivo_salida, index=False)

    n_final = len(df_combinado)
    print("====================================================")
    print("         REPORTE DE INTERPOLACIÓN COMPLETADO        ")
    print("====================================================")
    print(f"Filas originales (10s)         : {n_original}")
    print(f"Filas agregadas por interpolación: {total_interpolados}")
    print(f"Total de filas nuevo dataset (5s): {n_final}")
    print(f"Archivo guardado exitosamente en : {archivo_salida}")
    print("====================================================")

    # Validación rápida de diferencias temporales en el nuevo dataset
    diff_final = df_combinado["epoch_segundos"].diff()
    pct_5s = (diff_final.between(4.5, 5.5).sum() / (n_final - 1)) * 100
    print(f"Porcentaje de lecturas a ~5 segundos: {pct_5s:.2f}%")
    print("====================================================")

    return df_combinado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolar lecturas del dataset 1 (10s) a 5s para sincronizar con dataset 2."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="dataset_temperatura_crudo_1.csv",
        help="Archivo CSV original con lecturas cada 10s (por defecto: dataset_temperatura_crudo_1.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="dataset_temperatura_crudo_1_interpolado.csv",
        help="Archivo CSV de salida con lecturas interpoladas cada 5s (por defecto: dataset_temperatura_crudo_1_interpolado.csv)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=6.0,
        help="Delta mínimo en segundos para considerar interpolación (omite ráfagas de 0s)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=16.0,
        help="Delta máximo en segundos para considerar interpolación (omite pausas entre sesiones)",
    )
    parser.add_argument(
        "--todo",
        action="store_true",
        help="Forzar interpolación en absolutamente todos los pares consecutivos",
    )

    args = parser.parse_args()

    interpolar_temperatura_5s(
        archivo_entrada=args.input,
        archivo_salida=args.output,
        min_gap_segundos=args.min_gap,
        max_gap_segundos=args.max_gap,
        interpolar_todo=args.todo,
    )
