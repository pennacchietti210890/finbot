resource "kubernetes_deployment" "backend" {
  metadata {
    name = "finbot-backend"
    labels = {
      app = "finbot-backend"
    }
  }

  spec {
    replicas = 0  # Scale to zero by default

    selector {
      match_labels = {
        app = "finbot-backend"
      }
    }

    template {
      metadata {
        labels = {
          app = "finbot-backend"
        }
      }

      spec {
        container {
          name  = "backend"
          image = var.backend_image

          port {
            container_port = 8000
          }

        resources {
            requests = {
                cpu    = "500m"
                memory = "2Gi"
            }
            limits = {
                cpu    = "1"
                memory = "4Gi"
            }
        }

          env {
            name = "OPENAI_API_KEY"
            value_from {
              secret_key_ref {
                name = "finbot-secrets"
                key  = "OPENAI_API_KEY"
              }
            }
          }

          env {
            name = "FRED_API_KEY"
            value_from {
              secret_key_ref {
                name = "finbot-secrets"
                key  = "FRED_API_KEY"
              }
            }
          }

          env {
            name = "TAVILY_API_KEY"
            value_from {
              secret_key_ref {
                name = "finbot-secrets"
                key  = "TAVILY_API_KEY"
              }
            }
          }
          env {
            name  = "MCP_SERVER_URL"
           value_from {
              secret_key_ref {
                name = "finbot-secrets"
                key = "MCP_SERVER_URL"
              }
           }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "backend" {
  metadata {
    name = "finbot-backend"
  }

  spec {
    selector = {
      app = "finbot-backend"
    }

    port {
      port        = 8000
      target_port = 8000
    }

    type = "ClusterIP"
  }
}
