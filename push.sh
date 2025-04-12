#!/bin/bash

PROJECT="mattia-dev-2025"
REGION="europe-west1"
REPO="finbot"

declare -a SERVICES=("backend" "frontend" "mcp")

for SERVICE in "${SERVICES[@]}"
do
    echo "🔁 Tagging and pushing: $SERVICE"
    docker tag finbot-$SERVICE $REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:latest
    docker push $REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:latest
done
