from flask import Flask, request, render_template
from recommender import load_data, train_model, evaluate_model, get_recommendations

app = Flask(__name__)

ratings, movies = load_data()
training_data, test_data = ratings.randomSplit([0.8, 0.2])
model = train_model(training_data)
evaluate_model(model, test_data)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations, rated_movies, error = None, [], None  # <-- changed rated_movies to empty list
    if request.method == "POST":
        user_id = request.form.get("user_id")
        try:
            user_id = int(user_id)
            if not (0 <= user_id <= 943):
                error = "User ID must be between 0 and 943."
            else:
                recommendations, rated_movies = get_recommendations(model, ratings, movies, user_id)
        except ValueError:
            error = "Invalid user ID. Please enter a number between 0 and 943."
    return render_template("index.html", recommendations=recommendations, rated_movies=rated_movies, error=error)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
