output "dvc_bucket_name" {
  description = "DVC S3 bucket name"
  value       = aws_s3_bucket.dvc_bucket.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.repo.repository_url
}

output "ec2_public_ip" {
  description = "EC2 instance public IP"
  value       = aws_instance.llmops_ec2.public_ip
}

output "ec2_ssh_connection" {
  description = "SSH connection string"
  value       = "ssh ubuntu@${aws_instance.llmops_ec2.public_ip}"
} 