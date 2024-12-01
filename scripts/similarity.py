import pandas as pd
import unicodedata
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import time

class TextSimilarityPipeline:
    def __init__(self, df, text_column='ITE_ITEM_TITLE', n_results=None, language='portuguese'):
        """
        Inicializa el pipeline para calcular la similitud entre textos.
        
        Args:
            df (pd.DataFrame): DataFrame que contiene los textos a analizar.
            text_column (str): Nombre de la columna que contiene los textos.
            n_results (int, opcional): Número de resultados más similares a devolver.
            language (str): Idioma para eliminar palabras vacías (stop words).
        """
        self.df = df.copy()
        self.text_column = text_column
        self.n_results = n_results
        self.language = language
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

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
        Crea la matriz TF-IDF basada en los textos preprocesados.
        
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

    def calculate_similarities(self):
        """
        Calcula la similitud de coseno entre todos los pares de textos utilizando paralelización.
        
        Returns:
            self: Instancia actual con la tabla de similitud generada.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Debe ejecutar create_tfidf_matrix primero")
        
        # Calcular la matriz de similitudes
        cosine_sim = cosine_similarity(self.tfidf_matrix)
        # Generar combinaciones de índices (pares únicos)
        pairs = list(combinations(range(len(self.df)), 2))
        
        # Calcular las similitudes en paralelo
        results = Parallel(n_jobs=-1)(
            delayed(self.calculate_similarity_parallel)(pair, cosine_sim) 
            for pair in pairs
        )
        
        # Crear un DataFrame con los resultados
        self.similarity_df = pd.DataFrame(
            results,
            columns=['Término 1', 'Término 2', 'Similitud']
        )
        
        # Ordenar los resultados por similitud descendente
        self.similarity_df = self.similarity_df.sort_values(
            by='Similitud',
            ascending=False
        )
        
        # Seleccionar solo los n resultados más altos si se especifica
        if self.n_results:
            self.similarity_df = self.similarity_df.head(self.n_results)
        
        return self

    def run_pipeline(self):
        """
        Ejecuta todo el pipeline y devuelve los pares más similares.
        
        Returns:
            pd.DataFrame: DataFrame con las similitudes calculadas.
        """
        start_time = time.time()  # Iniciar temporizador
        self.create_tfidf_matrix()  # Crear la matriz TF-IDF
        self.calculate_similarities()  # Calcular similitudes
        print(f"Pipeline ejecutado en {time.time() - start_time:.2f} segundos")
        return self.similarity_df