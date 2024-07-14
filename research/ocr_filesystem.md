# OCR Filesystem Logic

## Requirements

- **Scalability and Speed**: The system needs to handle increasing loads efficiently and quickly,
  especially with a hopeful transition to video feeds in the future.
- **File Classification**: At some point the model needs to classify the type of document before processing it.
  This is important for speed/accuracy tradeoff.
- **OCR Processing**: Once classified, the document needs to be processed by the appropriate OCR service.
- **Minimize Latency**: Reduce the time taken to download and process the files.
- **Transition to Live Video Feeds**: Eventaully, the system should be capable of handling live video feed processing. Would take scan the page live.

## Idea Brainstorm

1. Use a centralized storage solution for my containers
    - Limits multiple downloads
    - Amazon S3 would be great for storing all files into a centralized, scalable solution
    - Gotta research further about the ins and outs of it

2. Event-Driven Architecture
    - Upload a file to S3 -> S3 triggers lambda function classification -> Lambda sends classification results back to SQS queue
    - OCR containers listen to queue and pick up tasks and process files

3. NEED to have efficient data transfer and processing
    - Pre-signed URLs: Use s3 urls to allow containers to access files without needing to redownload
    - Later down the road: can use kinesis for live video and real-time processing

4. Container orchestration
    - For sure pick up Kubernetes, need to practice this and it's probably the best at flexible scheduling which I'll need
        - could use docker swarm in the mean time to practice
    - Service Mesh: [Istio](https://istio.io/latest/about/service-mesh/)
        - Will be crucial for load balancing, and communication between services

5. Optimize File Classification
    - Inline CLassification: Classifier can be tied directly to lambda function so document type is immediately determined
    - Batch Processing: Batch files together rather than spawning new containers for every single file.

## Current Architecture plan

### 1. File Upload and Storage

**Flow:**

1. **User Uploads File to S3:**
   - Users upload files (PDFs, images, etc.) to a designated S3 bucket.
   - S3 provides scalable and durable storage, ensuring the files are safely stored and easily accessible.

2. **S3 Event Trigger:**
   - S3 triggers a Lambda function upon file upload (using S3 events).
   - The event includes metadata about the uploaded file (e.g., bucket name, file key).

**Implementation Steps:**

1. Create an S3 bucket.
2. Configure S3 bucket events to trigger a Lambda function on file upload.

### 2. File Classification

**Flow:**

1. **Lambda Function for Classification:**
   - The Lambda function retrieves the file from S3 using the information from the event.
   - It classifies the document type (e.g., invoice, receipt, report) using a pre-trained model or custom logic.
   - The classification result and file metadata are either stored in a DynamoDB table or sent to an SQS queue.

**Implementation Steps:**

1. Write a Lambda function to handle S3 events and classify the document.
2. Store classification results in DynamoDB or send a message to SQS.

### 3. OCR Processing

**Flow:**

1. **Polling SQS Queue:**
   - OCR containers, managed by ECS or Kubernetes, continuously poll the SQS queue for new tasks.
   - Each message in the queue contains the file’s S3 URL and classification result.

2. **Processing Files:**
   - OCR containers download the file from S3 using a pre-signed URL.
   - They process the file using the appropriate OCR model based on the classification result.
   - The OCR results are then stored back in S3 or a database for further use.

**Implementation Steps:**

1. Create an SQS queue.
2. Configure Lambda to send messages to the SQS queue after classification.
3. Set up ECS or Kubernetes to manage OCR containers.
4. Write OCR container code to poll SQS, process files, and store results.

### 4. Scaling and Orchestration

**Flow:**

1. **Container Orchestration:**
   - Use ECS or Kubernetes to deploy and manage OCR containers.
   - These platforms provide robust scaling capabilities to handle varying loads.

2. **Auto-Scaling:**
   - Configure auto-scaling based on SQS queue length or other metrics (e.g., CPU usage, memory usage).
   - This ensures that the number of OCR containers scales up during high load and scales down when idle.

**Implementation Steps:**

1. Set up ECS or Kubernetes cluster.
2. Deploy OCR containers to the cluster.
3. Configure auto-scaling policies based on relevant metrics.

### Detailed Breakdown of Components

#### File Upload and Storage

1. **S3 Bucket:**
   - Create an S3 bucket.
   - Configure bucket policies and permissions to allow Lambda and OCR containers access.

2. **S3 Event Trigger:**
   - Configure S3 to trigger a Lambda function on `s3:ObjectCreated:*` events.

#### File Classification

1. **Lambda Function:**
   - Write a Lambda function to handle S3 events:
     - Fetch the uploaded file from S3.
     - Perform classification (e.g., using a pre-trained ML model).
     - Store classification result in DynamoDB or send to SQS.

2. **DynamoDB Table (Optional):**
   - Create a DynamoDB table to store file metadata and classification results.

3. **SQS Queue:**
   - Create an SQS queue to receive classification results and file metadata.

#### OCR Processing

1. **OCR Container:**
   - Develop an OCR container application:
     - Poll SQS queue for new tasks.
     - Fetch the file from S3 using a pre-signed URL.
     - Process the file and perform OCR.
     - Store OCR results in S3 or a database.

2. **ECS/Kubernetes Cluster:**
   - Set up ECS or Kubernetes to manage OCR containers:
     - Define task definitions/pod specs.
     - Deploy the OCR container application.

#### Scaling and Orchestration

1. **Auto-Scaling:**
   - Configure ECS/Kubernetes auto-scaling:
     - Use CloudWatch metrics (e.g., SQS queue length, CPU/memory usage).
     - Define scaling policies to adjust the number of OCR containers dynamically.

2. **Monitoring and Logging:**
   - Set up monitoring  to track performance and health.
   - Configure logging (e.g., CloudWatch Logs, Elasticsearch) for troubleshooting and analysis.
