resource "kubernetes_ingress_v1" "frontend_ingress" {
  metadata {
    name = "finbot-ingress"
    annotations = {
        "kubernetes.io/ingress.class"                     = "gce"
        "networking.gke.io/managed-certificates"          = "finbot-cert"
        "networking.gke.io/enable-global-access"          = "true"
        "ingress.kubernetes.io/whitelist-source-range"    = var.allowed_ip
    }
  }

  spec {
    default_backend {
      service {
        name = kubernetes_service.frontend.metadata[0].name
        port {
          number = 80
        }
      }
    }
  }
}