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
	sleep 60
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

.PHONY: train train_monotone

train:
	python scripts/model.py

train_monotone:
	python scripts/model_monotone.py


.PHONY: start_api start_streamlit local_compose up_all stop_all \
        test_api_health test_api_metadata test_api_predict test_api_predict_file \
        build_streamlit_image build_api_image deploy_api deploy_all

start_api:
	@echo "Starting FastAPI (uvicorn) on port 8000..."
	@PORT=8000 uvicorn app.app:app --reload --port 8000


start_streamlit:
	@echo "Starting Streamlit on port 8501 (pointing to http://localhost:8000)..."
	@PREDICT_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py --server.port 8501

local_compose:
	@echo "Starting docker-compose services: db, api, streamlit..."
	docker compose up --build -d api streamlit

stop_all:
	@echo "Stopping api and streamlit services..."
	docker compose stop api streamlit || true
	docker compose rm -f api streamlit || true

up_all:
	@echo "Bringing up full stack (db, api, streamlit)..."
	docker compose up --build -d

HOST ?= localhost
PORT ?= 8000
API_URL := http://$(HOST):$(PORT)

test_api_health:
	@echo "Testing /health at $(API_URL)/health"
	@curl -sS $(API_URL)/health | python -m json.tool

test_api_metadata:
	@echo "Testing /metadata at $(API_URL)/metadata"
	@curl -sS $(API_URL)/metadata | python -m json.tool

test_api_predict:
	@echo "Testing /predict (inline sample payload) at $(API_URL)/predict"
	@curl -sS -X POST $(API_URL)/predict -H "Content-Type: application/json" \
		-d '{"rows":[{"neighbourhood_cleansed":"Louvre","property_type_slim":"Entire rental unit","room_type":"Entire home/apt","accommodates":2,"bedrooms":1,"beds":1,"bathrooms":1}]}' \
		| python -m json.tool

test_api_predict_file:
	@if [ -z "$(JSON)" ]; then echo "ERROR: set JSON=path/to/file.json"; exit 1; fi
	@echo "POSTing $(JSON) to $(API_URL)/predict"
	@curl -sS -X POST $(API_URL)/predict -H "Content-Type: application/json" -d @$${JSON} | python -m json.tool

build_streamlit_image:
	@echo "Building Streamlit Docker image..."
	docker build -f Dockerfile.streamlit -t $(DOCKER_IMAGE_NAME)-streamlit:local .

build_api_image:
	@echo "Building API Docker image (wrapper target)..."
	$(MAKE) docker_build

deploy_api: docker_allow docker_build docker_push docker_deploy
	@echo "API deployed (docker_build -> docker_push -> docker_deploy finished)."

deploy_all: deploy_api
	@echo "Deploy completed for API. If you want to deploy Streamlit as well, build/push that image and create a Cloud Run service for it."
