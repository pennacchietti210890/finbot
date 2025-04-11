resource "kubernetes_deployment" "frontend" {
  metadata {
    name = "finbot-frontend"
    labels = {
      app = "finbot-frontend"
    }
  }

  spec {
    replicas = 0  # Default to 0 to avoid idle cost
    selector {
      match_labels = {
        app = "finbot-frontend"
      }
    }

    template {
      metadata {
        labels = {
          app = "finbot-frontend"
        }
      }

      spec {
        container {
          name  = "frontend"
          image = var.frontend_image

          port {
            container_port = 8502 
          }

          env {
            name  = "API_URL"
            value = "http://finbot-backend:8000"
        }

        resources {
            limits = {
                cpu    = "250m"
                memory = "256Mi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "frontend" {
  metadata {
    name = "finbot-frontend"
    labels = {
      app = "finbot-frontend"
    }
  }

  spec {
    selector = {
      app = "finbot-frontend"
    }

    port {
      port        = 80
      target_port = 8502  # Match container port above
    }

    type = "NodePort"  # Will be exposed via Ingress
  }
}
