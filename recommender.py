from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
import requests
from urllib.parse import quote_plus

spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .getOrCreate()

TMDB_API_KEY = "cb4c6d2ddbebbebda57b6aa087f01b21"

def fetch_movie_banner_and_trailer(title):
    

    query = title.split(' (')[0]  # Remove year for better TMDb match
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    response = requests.get(search_url)
    data = response.json()

    banner_url, trailer_url = None, None

    if data['results']:
        movie_id = data['results'][0]['id']
        poster_path = data['results'][0].get('poster_path')
        banner_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None

        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        videos_response = requests.get(videos_url)
        videos = videos_response.json().get("results", [])

        # Try trailer first
        for v in videos:
            if v.get("type") == "Trailer" and v.get("site") == "YouTube":
                trailer_url = f"https://www.youtube.com/watch?v={v['key']}"
                break

        # Fallback: any YouTube video
        if not trailer_url:
            for v in videos:
                if v.get("site") == "YouTube":
                    trailer_url = f"https://www.youtube.com/watch?v={v['key']}"
                    break

    # Final fallback: YouTube search link
    if not trailer_url:
        trailer_url = f"https://www.youtube.com/results?search_query={quote_plus(title + ' trailer')}"

    return banner_url, trailer_url



def load_data():
    ratings = spark.read.csv("data/u.data", sep="\t", inferSchema=True)
    ratings = ratings.toDF("user_id", "item_id", "rating", "timestamp")
    ratings = ratings.dropDuplicates().dropna()

    movies = spark.read.csv("data/u.item", sep="|", inferSchema=True, encoding="ISO-8859-1")
    movies = movies.toDF("item_id", "title", *["col" + str(i) for i in range(22)]) \
                   .select("item_id", "title")
    movies = movies.dropDuplicates().dropna()

    return ratings, movies

def train_model(train_data):
    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    return als.fit(train_data)

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) = {rmse:.4f}")

def get_recommendations(model, ratings, movies, user_id):
    user_df = ratings.select("user_id").distinct().filter(col("user_id") == user_id)
    user_rated = ratings.filter(col("user_id") == user_id).join(movies, on="item_id").select("title", "rating")

    user_recs = model.recommendForUserSubset(user_df, 10)
    if user_recs.count() == 0:
        return [], [{"title": row["title"], "rating": row["rating"]} for row in user_rated.collect()]

    recs = user_recs.selectExpr("explode(recommendations) as rec").selectExpr("rec.item_id", "rec.rating")
    recs_df = recs.join(movies, on="item_id").select("title", "rating")

    results = []
    for row in recs_df.collect():
        banner, trailer = fetch_movie_banner_and_trailer(row["title"])
        results.append({
            "title": row["title"],
            "predicted_rating": round(row["rating"], 2),
            "banner": banner,
            "trailer": trailer
        })

    rated = [{"title": row["title"], "rating": row["rating"]} for row in user_rated.collect()]
    return results, rated
