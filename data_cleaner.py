import csv
import os
import sqlite3
import numpy as np

DB_PATH = "nano_ai_sensor.db"
CSV_ENTRADA = "datos_arduino.csv" # NOMBRE DEL CSV

# CREAR ESQUEMA SQLITE CON LAS COLUMNAS EN ESPAÑOL
def inicializar_bd(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    # TABLA 1: TODO EL CSV TAL CUAL ENTRA (con errores y todo)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_iso TEXT,
            epoch_segundos REAL,
            fecha TEXT,
            hora_decimal REAL,
            lectura_serial_cruda TEXT,
            temp_c REAL,
            es_perturbacion_manual INTEGER
        );
    """)

    # TABLA 2: SOLO LECTURAS SIN FALLOS FISICOS (incluye normales=0 y perturbaciones=1)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clean_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER,
            timestamp_iso TEXT NOT NULL,
            epoch_segundos REAL NOT NULL,
            fecha TEXT NOT NULL,
            hora_decimal REAL NOT NULL,
            lectura_serial_cruda TEXT NOT NULL,
            temp_c REAL NOT NULL,
            es_perturbacion_manual INTEGER NOT NULL,
            FOREIGN KEY (raw_id) REFERENCES raw_readings(id)
        );
    """)

    # TABLA 3: AUDITORÍA DE ERRORES DE HARDWARE (para defender en el Demo)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER,
            timestamp_iso TEXT,
            lectura_serial_cruda TEXT,
            valor_detectado REAL,
            motivo_rechazo TEXT NOT NULL,
            FOREIGN KEY (raw_id) REFERENCES raw_readings(id)
        );
    """)

    conn.commit()
    conn.close()

# CARGAR CSV, LIMPIAR CON NUMPY Y GUARDAR EN BD
def procesar_csv_con_numpy(csv_path=CSV_ENTRADA, db_path=DB_PATH):
    # LEER EL CSV
    filas = [] 
    with open(csv_path, mode="r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            filas.append(fila)

    if not filas:
        print("El archivo CSV está vacío.")
        return

    n = len(filas)
    # EXTRAER ARREGLOS DE NUMPY PARA HACER LA LIMPIEZA VECTORIZADA (SIN FOR-LOOPS LENTOS)
    temp_c_arr = np.empty(n, dtype=np.float64)
    perturbacion_arr = np.zeros(n, dtype=np.int32)
    serial_ok_arr = np.zeros(n, dtype=bool)

    for i, f in enumerate(filas):
        # VALIDAR SI LA CADENA SERIAL TIENE EL FORMATO CORRECTO "TEMP:XX.XXXX" (equivalente al CRC=YES en Arduino)
        crudo = str(f["lectura_serial_cruda"]).strip()
        serial_ok_arr[i] = crudo.startswith("TEMP:")

        try:
            temp_c_arr[i] = float(f["temp_c"])
        except (ValueError, TypeError):
            temp_c_arr[i] = np.nan

        try:
            perturbacion_arr[i] = int(f["es_perturbacion_manual"])
        except (ValueError, TypeError):
            perturbacion_arr[i] = 0

    # --- MÁSCARAS BOOLEANAS DE NUMPY (Filtros de Hardware DS18B20) ---
    mask_nan = np.isnan(temp_c_arr)
    mask_serial_mal = ~serial_ok_arr
    mask_error_127 = np.isclose(temp_c_arr, -127.0, atol=0.1)  # Cable desconectado
    mask_error_85 = np.isclose(temp_c_arr, 85.0, atol=0.1)     # Reset de fábrica DS18B20

    # Un dato es VÁLIDO de hardware si no tiene ninguno de los 4 errores anteriores
    mask_hardware_valido = ~(mask_nan | mask_serial_mal | mask_error_127 | mask_error_85)

    # Separación lógica con NumPy para los modelos de IA:
    # 1) Datos para ENTRENAR (Hardware válido Y sin perturbación manual: es_perturbacion_manual == 0)
    mask_entrenamiento_normal = mask_hardware_valido & (perturbacion_arr == 0)
    # 2) Datos de ANOMALÍAS REALES (Hardware válido Y con hielo/calor: es_perturbacion_manual == 1)
    mask_anomalias_reales = mask_hardware_valido & (perturbacion_arr == 1)

    # --- GUARDAR EN SQLITE ---
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for i, f in enumerate(filas):
        t_val = None if mask_nan[i] else float(temp_c_arr[i])
        p_val = int(perturbacion_arr[i])

        # 1. Insertar SIEMPRE en raw_readings
        cursor.execute("""
            INSERT INTO raw_readings 
            (timestamp_iso, epoch_segundos, fecha, hora_decimal, lectura_serial_cruda, temp_c, es_perturbacion_manual)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f["timestamp_iso"], float(f["epoch_segundos"]), f["fecha"],
            float(f["hora_decimal"]), f["lectura_serial_cruda"], t_val, p_val
        ))
        raw_id = cursor.lastrowid

        # 2. Si pasó el filtro de NumPy -> va a clean_readings
        if mask_hardware_valido[i]:
            cursor.execute("""
                INSERT INTO clean_readings 
                (raw_id, timestamp_iso, epoch_segundos, fecha, hora_decimal, lectura_serial_cruda, temp_c, es_perturbacion_manual)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                raw_id, f["timestamp_iso"], float(f["epoch_segundos"]), f["fecha"],
                float(f["hora_decimal"]), f["lectura_serial_cruda"], t_val, p_val
            ))
        # 3. Si falló el filtro de NumPy -> registrar la causa exacta en logs
        else:
            if mask_serial_mal[i]:
                motivo = "Error de trama serial (No inicia con TEMP: / CRC inválido)"
            elif mask_nan[i]:
                motivo = "Valor nulo o corrupto (NaN)"
            elif mask_error_127[i]:
                motivo = "Fallo físico DS18B20: Desconexión (-127.0 °C)"
            elif mask_error_85[i]:
                motivo = "Fallo físico DS18B20: Power-on reset (85.0 °C)"
            else:
                motivo = "Error desconocido"

            cursor.execute("""
                INSERT INTO logs (raw_id, timestamp_iso, lectura_serial_cruda, valor_detectado, motivo_rechazo)
                VALUES (?, ?, ?, ?, ?)
            """, (raw_id, f["timestamp_iso"], f["lectura_serial_cruda"], t_val, motivo))

    conn.commit()
    conn.close()

    # --- EXPORTAR CSVs LIMPIOS PARA EL MODEL LEAD ---
    campos = [
        "timestamp_iso", "epoch_segundos", "fecha", "hora_decimal",
        "lectura_serial_cruda", "temp_c", "es_perturbacion_manual"
    ]
    
    # CSV 1: Todo limpio de fallos físicos (para el historial de la BD)
    guardar_subconjunto_csv("clean_completo.csv", filas, mask_hardware_valido, campos)
    # CSV 2: SOLO NORMALES (es_perturbacion_manual == 0) -> Para entrenar el Autoencoder y Regresor
    guardar_subconjunto_csv("train_solo_normales.csv", filas, mask_entrenamiento_normal, campos)

    # --- REPORTE EN CONSOLA PARA TU BITÁCORA ---
    print("====================================================")
    print("         REPORTE DE LIMPIEZA NUMPY (DATA LEAD)      ")
    print("====================================================")
    print(f"Total de filas leídas del CSV : {n}")
    print(f"Lecturas limpias (Válidas)    : {np.sum(mask_hardware_valido)}")
    print(f"  ├─ Normales (perturbación=0): {np.sum(mask_entrenamiento_normal)} -> (Usar para Entrenar IA)")
    print(f"  └─ Anomalías (perturbación=1): {np.sum(mask_anomalias_reales)} -> (Usar para Test de Anomalías)")
    print(f"Lecturas basura eliminadas    : {np.sum(~mask_hardware_valido)}")
    print(f"  ├─ Por -127.0 °C (Cable)    : {np.sum(mask_error_127)}")
    print(f"  ├─ Por   85.0 °C (Reset)    : {np.sum(mask_error_85)}")
    print(f"  └─ Por Trama Serial/NaN     : {np.sum(mask_serial_mal | mask_nan)}")
    print("====================================================")


def guardar_subconjunto_csv(nombre_archivo, filas_originales, mascara_booleana, campos):
    indices_validos = np.where(mascara_booleana)[0]
    with open(nombre_archivo, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for idx in indices_validos:
            writer.writerow(filas_originales[idx])


# ==========================================
# BLOQUE DE PRUEBA (Con los datos exactos de tu captura de pantalla)
# ==========================================
if __name__ == "__main__":
    # Si aún no tienes el archivo CSV en la carpeta, este bloque crea uno igual al de tu foto para probar
    if not os.path.exists(CSV_ENTRADA):
        datos_foto = [
            ["timestamp_iso", "epoch_segundos", "fecha", "hora_decimal", "lectura_serial_cruda", "temp_c", "es_perturbacion_manual"],
            ["2026-10-20T18:00:00", "1792540800.12", "2026-10-20", "18.0000", "TEMP:24.5625", "24.5625", "0"],
            ["2026-10-20T18:00:10", "1792540810.12", "2026-10-20", "18.0028", "TEMP:24.5625", "24.5625", "0"],
            ["2026-10-20T18:00:20", "1792540820.13", "2026-10-20", "18.0056", "TEMP:24.6250", "24.6250", "0"],
            ["2026-10-20T18:00:30", "1792540830.13", "2026-10-20", "18.0083", "TEMP:-127.0000", "-127.0", "0"],
            ["2026-10-21T12:30:00", "1792607400.45", "2026-10-21", "12.5000", "TEMP:25.1250", "25.1250", "0"],
            ["2026-10-22T16:15:00", "1792707300.80", "2026-10-22", "16.2500", "TEMP:29.8750", "29.8750", "1"],
            ["2026-10-22T16:15:10", "1792707310.80", "2026-10-22", "16.2528", "TEMP:31.4375", "31.4375", "1"],
        ]
        with open(CSV_ENTRADA, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(datos_foto)
        print(f"Archivo de prueba '{CSV_ENTRADA}' generado con los datos de tu imagen.\n")

    inicializar_bd()
    procesar_csv_con_numpy()