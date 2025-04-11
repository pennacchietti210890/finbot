resource "google_container_cluster" "gke" {
  name     = "finbot-cluster"
  location = var.region

  enable_autopilot = true

  network    = "default"
  subnetwork = "default"

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = null
    services_secondary_range_name = null
  }
}
