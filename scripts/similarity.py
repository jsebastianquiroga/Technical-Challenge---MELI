import pandas as pd
import unicodedata
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from itertools import combinations
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import time
from tqdm import tqdm


class TextSimilarityPipeline:
    def __init__(self, df, text_column='ITE_ITEM_TITLE', n_results=None, language='portuguese', n_components=100):
        """
        Inicializa el pipeline para calcular la similitud entre textos con reducción de dimensionalidad.
        
        Args:
            df (pd.DataFrame): DataFrame que contiene los textos a analizar.
            text_column (str): Nombre de la columna que contiene los textos.
            n_results (int, opcional): Número de resultados más similares a devolver.
            language (str): Idioma para eliminar palabras vacías (stop words).
            n_components (int): Número de dimensiones para reducir con SVD.
        """
        self.df = df.copy()
        self.text_column = text_column
        self.n_results = n_results
        self.language = language
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.reduced_matrix = None
        self.similarity_df = None

        # Cargar palabras vacías para el idioma especificado
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))

    def preprocess_text(self, text):
        """
        Limpia y normaliza un texto:
        - Convierte a minúsculas.
        - Elimina acentos.
        
        Args:
            text (str): Texto a limpiar.
            
        Returns:
            str: Texto normalizado.
        """
        text = text.lower()  # Convertir a minúsculas
        text = ''.join(
            char for char in unicodedata.normalize('NFKD', text) 
            if not unicodedata.combining(char)  # Eliminar acentos
        )
        return text

    def tokenize_and_clean(self, text):
        """
        Tokeniza un texto y elimina palabras vacías (stop words).
        
        Args:
            text (str): Texto a tokenizar.
            
        Returns:
            list: Lista de palabras limpias.
        """
        tokens = text.split()  # Tokenizar por espacios
        return [word for word in tokens if word not in self.stop_words]

    def create_tfidf_matrix(self):
        """
        Crea la matriz TF-IDF basada en los textos preprocesados y aplica SVD para reducir la dimensionalidad.
        
        Returns:
            self: Instancia actual para encadenar métodos.
        """
        # Preprocesar el texto
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.preprocess_text)
        # Tokenizar y eliminar palabras vacías
        self.df['tokens'] = self.df['cleaned_text'].apply(self.tokenize_and_clean)
        # Unir tokens para formar el texto final
        processed_texts = self.df['tokens'].apply(' '.join)
        # Crear la matriz TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        # Reducir dimensionalidad con Truncated SVD
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.reduced_matrix = svd.fit_transform(self.tfidf_matrix)
        return self

    def calculate_similarity_parallel(self, pair, cosine_sim):
        """
        Calcula la similitud de coseno para un par de índices.

        Args:
            pair (tuple): Tupla con los índices de los textos a comparar.
            cosine_sim (ndarray): Matriz de similitud de coseno.

        Returns:
            list: Lista con los textos y su puntaje de similitud.
        """
        i, j = pair
        title1 = self.df.iloc[i][self.text_column]
        title2 = self.df.iloc[j][self.text_column]
        score = cosine_sim[i, j]
        return [title1, title2, score]

    def calculate_similarities(self, batch_size=500, threshold= 0.5):
        """
        Calcula la similitud de coseno entre todos los pares de textos utilizando 
        procesamiento por lotes y filtrado por umbral.
        
        Args:
            batch_size (int): Tamaño del lote para procesar
            threshold (float): Umbral mínimo de similitud para guardar
            
        Returns:
            self: Instancia actual con la tabla de similitud generada.
        """
        if self.reduced_matrix is None:
            raise ValueError("Debe ejecutar create_tfidf_matrix primero")
        
        start_time = time.time()
        n_samples = len(self.df)
        results = []
        
        print("Calculando similitudes por lotes...")
        for i in tqdm(range(0, n_samples, batch_size)):
            batch_end = min(i + batch_size, n_samples)
            # Calcular similitud solo para el lote actual
            batch_similarities = cosine_similarity(
                self.reduced_matrix[i:batch_end],
                self.reduced_matrix
            )
            
            # Procesar solo similitudes sobre el umbral
            for batch_idx, row in enumerate(batch_similarities):
                abs_idx = i + batch_idx
                # Solo procesar la mitad superior de la matriz
                high_similarities = np.where(row[abs_idx+1:] > threshold)[0]
                
                for j in high_similarities:
                    j = j + abs_idx + 1  # Ajustar índice
                    results.append([
                        self.df.iloc[abs_idx][self.text_column],
                        self.df.iloc[j][self.text_column],
                        row[j]
                    ])
        
        # Crear DataFrame con resultados filtrados
        self.similarity_df = pd.DataFrame(
            results,
            columns=['Término 1', 'Término 2', 'Similitud']
        ).sort_values('Similitud', ascending=False)
        
        # Seleccionar solo los n resultados más altos si se especifica
        if self.n_results:
            self.similarity_df = self.similarity_df.head(self.n_results)
        
        calculation_time = time.time() - start_time
        print(f"Cálculo por lotes completado en {calculation_time:.2f} segundos")
        
        return self

    def run_pipeline(self):
        """
        Ejecuta todo el pipeline y devuelve los pares más similares.
        
        Returns:
            pd.DataFrame: DataFrame con las similitudes calculadas.
        """
        start_time = time.time()  # Iniciar temporizador
        self.create_tfidf_matrix()  # Crear la matriz TF-IDF con reducción
        self.calculate_similarities()  # Calcular similitudes
        total_time = time.time() - start_time
        print(f"Pipeline ejecutado en {total_time:.2f} segundos")
        return self.similarity_df