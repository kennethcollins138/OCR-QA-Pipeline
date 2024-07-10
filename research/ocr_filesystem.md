# OCR Filesystem Logic

## Requirements

- **Scalability and Speed**: The system needs to handle increasing loads efficiently and quickly,
  especially with a hopeful transition to video feeds in the future.
- **File Classification**: At some point the model needs to classify the type of document before processing it.
  This is important for speed/accuracy tradeoff.
- **OCR Processing**: Once classified, the document needs to be processed by the appropriate OCR service.
- **Minimize Latency**: Reduce the time taken to download and process the files.
- **Transition to Live Video Feeds**: Eventaully, the system should be capable of handling live video feed processing.

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

- 
