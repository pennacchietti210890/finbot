variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "mattia-dev-2025"
}

variable "region" {
  description = "GCP region for GKE and Artifact Registry"
  type        = string
  default     = "europe-west1"
}

variable "frontend_image" {
  description = "Artifact Registry image for frontend"
  type        = string
  default     = "europe-west1-docker.pkg.dev/mattia-dev-2025/finbot/frontend:latest"
}

variable "backend_image" {
  description = "Artifact Registry image for backend"
  type        = string
  default     = "europe-west1-docker.pkg.dev/mattia-dev-2025/finbot/backend:latest"
}

variable "allowed_ip" {
  description = "Your personal IP in CIDR format (e.g., 123.45.67.89/32)"
  type        = string
}

variable "mcp_image" {
  description = "Artifact Registry image for MCP server"
  type        = string
  default     = "europe-west1-docker.pkg.dev/mattia-dev-2025/finbot/mcp:latest"
}