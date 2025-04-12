resource "kubernetes_deployment" "mcp" {
  metadata {
    name = "finbot-mcp"
    labels = {
      app = "finbot-mcp"
    }
  }

  spec {
    replicas = 0

    selector {
      match_labels = {
        app = "finbot-mcp"
      }
    }

    template {
      metadata {
        labels = {
          app = "finbot-mcp"
        }
      }

      spec {
        container {
          name  = "mcp"
          image = var.mcp_image

          port {
            container_port = 5005
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
           limits = {
              cpu    = "500m"
              memory = "1Gi"
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
 
          }
        }
      }
    }
  }


resource "kubernetes_service" "mcp" {
  metadata {
    name = "finbot-mcp"
  }

  spec {
    selector = {
      app = "finbot-mcp"
    }

    port {
      port        = 5005
      target_port = 5005
    }

    type = "ClusterIP"
  }
}
