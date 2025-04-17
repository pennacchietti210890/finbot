# Target: dependencies
push:
	./scripts/push.sh

# Usage: make tf_build IP=101.57.38.236/32
IP ?= 127.0.0.1/32

tf_build:
	cd infra && terraform apply -var="allowed_ip=$(IP)" -auto-approve --lock=false

clusters_scale_up:
	kubectl scale deployment finbot-frontend --replicas=1
	kubectl scale deployment finbot-backend --replicas=1
	kubectl scale deployment finbot-mcp --replicas=1

clusters_scale_down:
	kubectl scale deployment finbot-frontend --replicas=0
	kubectl scale deployment finbot-backend --replicas=0
	kubectl scale deployment finbot-mcp --replicas=0