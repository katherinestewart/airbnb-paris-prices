.PHONY: up schema load indexes db verify down reset

up:
	docker compose up -d

schema: up
	docker compose exec -T db psql -U airbnb -d airbnb -f /work/sql/schema.sql

load:
	docker compose exec -T db psql -U airbnb -d airbnb -f /work/sql/load.sql

indexes:
	docker compose exec -T db psql -U airbnb -d airbnb -f /work/sql/indexes.sql

db: up schema load indexes

verify:
	docker compose exec -T db psql -U airbnb -d airbnb -c "\dt raw.*"
	docker compose exec -T db psql -U airbnb -d airbnb -c "SELECT COUNT(*) FROM raw.listings;"
	docker compose exec -T db psql -U airbnb -d airbnb -c "SELECT COUNT(*) FROM raw.reviews;"
	docker compose exec -T db psql -U airbnb -d airbnb -c "\d raw.listings"
	docker compose exec -T db psql -U airbnb -d airbnb -c "\d raw.reviews"

down:
	docker compose down

reset:
	docker compose down -v
	docker compose up -d
	sleep 5
	docker compose exec -T db psql -U airbnb -d airbnb -f /work/sql/schema.sql
	docker compose exec -T db psql -U airbnb -d airbnb -f /work/sql/load.sql
	docker compose exec -T db psql -U airbnb -d airbnb -f /work/sql/indexes.sql

count_rows:
	docker compose exec -T db psql -U airbnb -d airbnb -c "\
	SELECT 'listings' AS table, COUNT(*) AS n FROM raw.listings \
	UNION ALL \
	SELECT 'reviews', COUNT(*) FROM raw.reviews;"

clean:
	python scripts/clean_listings.py
	python scripts/clean_reviews.py

verify_clean:
	@{ \
	  echo "table|cols|rows|min_price|max_price"; \
	  docker compose exec -T db psql -U airbnb -d airbnb -X -qAt -F '|' -c "\
	WITH cols AS ( \
		SELECT table_name, COUNT(*) AS cols \
		FROM information_schema.columns \
		WHERE table_schema='clean' \
		GROUP BY table_name \
	) \
	SELECT 'listings_features', \
				(SELECT cols FROM cols WHERE table_name='listings_features'), \
				COUNT(*), MIN(price), MAX(price) \
	FROM clean.listings_features \
	UNION ALL \
	SELECT 'reviews', \
				(SELECT cols FROM cols WHERE table_name='reviews'), \
				COUNT(*), NULL::double precision, NULL::double precision \
	FROM clean.reviews \
	UNION ALL \
	SELECT 'reviews_summary', \
				(SELECT cols FROM cols WHERE table_name='reviews_summary'), \
				COUNT(*), NULL::double precision, NULL::double precision \
	FROM clean.reviews_summary;"; \
		} | column -t -s '|'

.PHONY: cache train app

cache:
	python -c "from src.data.load_data import load_data; load_data(use_cache=True)"

train:
	python src/models/train.py

evaluate:
	python src/models/evaluate.py

app:
	streamlit run app/streamlit_app.py
