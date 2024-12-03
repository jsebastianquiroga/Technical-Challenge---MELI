import pandas as pd
import numpy as np
import chardet
import os
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Inicializa el preprocesador de datos.
        
        Args:
            file_path (str): Ruta al archivo CSV.
        """
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.df_with_features = None
        self.original_stats = {}
        self.processed_stats = {}

    def detect_encoding(self):
        """
        Detecta la codificación del archivo CSV.
        
        Returns:
            str: Codificación detectada.
        """
        with open(self.file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
        return result['encoding']

    def load_data(self):
        """
        Carga los datos del archivo CSV y muestra estadísticas iniciales.
        """
        encoding = self.detect_encoding()
        self.df = pd.read_csv(self.file_path, encoding=encoding)
        
        # Guardar estadísticas iniciales
        self.original_stats['total_records'] = len(self.df)
        self.original_stats['total_devices'] = self.df['device'].nunique()
        
        if 'failure' in self.df.columns:
            print('\nDistribución inicial de failure:')
            print(self.df.failure.value_counts())
            print('\nPorcentajes:')
            print(self.df.failure.value_counts(normalize=True))
        
        print(f"\nRegistros iniciales: {self.original_stats['total_records']}")
        print(f"Dispositivos iniciales: {self.original_stats['total_devices']}")
        print("Datos cargados exitosamente.")

    def preprocess_data(self):
        """
        Realiza la limpieza inicial y transformación de fechas en el dataset.
        """
        df = self.df.copy()
        initial_records = len(df)
        initial_devices = df['device'].nunique()

        # Eliminar duplicados
        df = df.drop_duplicates()
        records_after_duplicates = len(df)
        print(f"\nDuplicados eliminados: {initial_records - records_after_duplicates}")
        print(f"Registros después de eliminar duplicados: {records_after_duplicates}")

        # Revisar valores nulos
        null_summary = df.isnull().sum()
        print("\nValores nulos por columna:")
        print(null_summary[null_summary > 0])

        # Imputación de valores nulos
        for col in df.columns:
            if col.startswith("attribute"):
                nulls_before = df[col].isnull().sum()
                df[col].fillna(df[col].median(), inplace=True)
                # print(f"Valores nulos imputados en {col}: {nulls_before}")

        # Convertir la columna 'date' a tipo datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df['date'].isnull().sum()
        if invalid_dates > 0:
            print(f"\nFechas inválidas encontradas: {invalid_dates}")
        
        df = df.sort_values(by=['device', 'date'])

        # Revisar periodicidad y eliminar dispositivos con gaps grandes
        df['days_diff'] = df.groupby('device')['date'].diff().dt.days
        print("\nEstadísticas de días entre registros:")
        print(df['days_diff'].describe())
        
        invalid_devices = df[df['days_diff'] > 30]['device'].unique()
        
        if 'failure' in df.columns:
            critical_devices = set(invalid_devices) & set(df[df['failure'] == 1]['device'].unique())
            print(f"\nDispositivos con fallas que serán eliminados: {len(critical_devices)}")
        
        df = df[~df['device'].isin(invalid_devices)]
        print(f"Dispositivos con gaps eliminados: {len(invalid_devices)}")

        df.reset_index(drop=True, inplace=True)
        self.df_cleaned = df
        
        # Guardar estadísticas procesadas
        self.processed_stats['total_records'] = len(df)
        self.processed_stats['total_devices'] = df['device'].nunique()
        
        print("\nResumen de cambios:")
        print(f"Registros iniciales: {initial_records}")
        print(f"Registros finales: {len(df)}")
        print(f"Reducción: {((initial_records - len(df)) / initial_records) * 100:.2f}%")
        
        if 'failure' in df.columns:
            print("\nDistribución final de failure:")
            print(df.failure.value_counts())
            print("\nPorcentajes finales:")
            print(df.failure.value_counts(normalize=True))
        
        print("\nPreprocesamiento de datos completado.")

    def generate_temporal_features(self, attributes, lags=[1, 2, 3], rolling_windows=[3, 5]):
        """
        Genera características temporales como lags, medias móviles, diferencias y tasas de cambio.
        
        Args:
            attributes (list): Lista de atributos para los cuales generar características.
            lags (list): Lista de valores de lag a considerar.
            rolling_windows (list): Ventanas para medias móviles.
        """
        df = self.df_cleaned.copy()
        initial_columns = len(df.columns)
        df = df.sort_values(by=['device', 'date'])

        for attr in attributes:
            print(f"\nGenerando características para: {attr}")
            
            for lag in lags:
                df[f'{attr}_lag{lag}'] = df.groupby('device')[attr].shift(lag)
                
            for window in rolling_windows:
                df[f'{attr}_rolling_mean_{window}'] = (
                    df.groupby('device')[attr]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
            df[f'{attr}_diff'] = df[attr] - df.groupby('device')[attr].shift(1)
            df[f'{attr}_rate_of_change'] = (
                df[f'{attr}_diff'] / df.groupby('device')[attr].shift(1)
            )

        # Imputar valores NaN con estrategia específica
        temporal_columns = [col for col in df.columns if any(x in col for x in ['lag', 'rolling', 'diff', 'rate_of_change'])]
        
        for col in temporal_columns:
            nulls_before = df[col].isnull().sum()
            if 'rate_of_change' in col:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(0, inplace=True)
            print(f"Valores nulos imputados en {col}: {nulls_before}")

        self.df_with_features = df
        print(f"\nCaracterísticas generadas: {len(df.columns) - initial_columns}")
        print("Características temporales generadas.")

    def save_processed_data(self, output_path='../data/processed_data.csv'):
        """
        Guarda el dataset procesado en un archivo CSV.
        
        Args:
            output_path (str): Ruta para guardar el archivo procesado.
        """
        if self.df_with_features is not None:
            self.df_with_features.to_csv(output_path, index=False)
            print(f"Datos procesados guardados en {output_path}")
            print(f"Dimensiones del archivo guardado: {self.df_with_features.shape}")
        else:
            print("Error: No hay datos procesados para guardar.")

    def run_pipeline(self):
        """
        Ejecuta todo el pipeline de preprocesamiento y generación de características.
        """
        print("Iniciando pipeline de procesamiento...")
        self.load_data()
        self.preprocess_data()
        attributes = [col for col in self.df_cleaned.columns if col.startswith('attribute')]
        self.generate_temporal_features(attributes)
        print("\nPipeline completado.")
        print(f"Dimensiones finales del dataset: {self.df_with_features.shape}")