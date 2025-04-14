# Target: dependencies
push:
	./scripts/push.sh

# Usage: make tf_build IP=101.57.38.236/32
IP ?= 127.0.0.1/32

tf_build:
	cd infra && terraform apply -var="allowed_ip=$(IP)" -auto-approve --lock=false
