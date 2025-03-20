-- Insert test user
INSERT INTO users (email, password_hash, role) 
VALUES ('test@example.com', '$2a$10$6lP.jFgqPqF4sfqmKviMyewxnGXGLHYVFrh5cL3LQtXXNKvOXu3Uy', 'admin')
ON CONFLICT (email) DO NOTHING;

-- Insert test user for nilson.peres@usp.br
INSERT INTO users (email, password_hash, role) 
VALUES ('nilson.peres@usp.br', '$2a$10$6lP.jFgqPqF4sfqmKviMyewxnGXGLHYVFrh5cL3LQtXXNKvOXu3Uy', 'admin')
ON CONFLICT (email) DO NOTHING;

-- Insert test locations with all columns
INSERT INTO locations (name, description, latitude, longitude, person_count, vehicle_count, avg_flow, last_updated) VALUES
('São Paulo', 'Main office in São Paulo', -23.5505, -46.6333, 150, 75, 225.0, CURRENT_TIMESTAMP),
('Rio de Janeiro', 'Branch office in Rio', -22.9068, -43.1729, 120, 60, 180.0, CURRENT_TIMESTAMP),
('Belo Horizonte', 'Branch office in BH', -19.9167, -43.9345, 100, 50, 150.0, CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;

-- Insert test models
INSERT INTO models (name, version, type, framework, status, cloud_platform, endpoint_url, accuracy, avg_inference_time, created_by, deployed_at)
SELECT 
    'YOLOv5', 'v1.0', 'object_detection', 'PyTorch', 'deployed', 'AWS', 'https://api.example.com/yolov5', 0.95, 0.15, id, CURRENT_TIMESTAMP
FROM users WHERE email = 'test@example.com'
ON CONFLICT DO NOTHING;

INSERT INTO models (name, version, type, framework, status, cloud_platform, endpoint_url, accuracy, avg_inference_time, created_by, deployed_at)
SELECT 
    'Faster R-CNN', 'v2.0', 'object_detection', 'TensorFlow', 'deployed', 'Azure', 'https://api.example.com/faster-rcnn', 0.92, 0.25, id, CURRENT_TIMESTAMP
FROM users WHERE email = 'test@example.com'
ON CONFLICT DO NOTHING;

-- Insert test models for nilson.peres@usp.br
INSERT INTO models (name, version, type, framework, status, cloud_platform, endpoint_url, accuracy, avg_inference_time, created_by, deployed_at)
SELECT 
    'YOLOv8', 'v1.0', 'object_detection', 'PyTorch', 'deployed', 'AWS', 'https://api.example.com/yolov8', 0.97, 0.12, id, CURRENT_TIMESTAMP
FROM users WHERE email = 'nilson.peres@usp.br'
ON CONFLICT DO NOTHING;

INSERT INTO models (name, version, type, framework, status, cloud_platform, endpoint_url, accuracy, avg_inference_time, created_by, deployed_at)
SELECT 
    'DETR', 'v1.0', 'object_detection', 'PyTorch', 'deployed', 'Azure', 'https://api.example.com/detr', 0.94, 0.28, id, CURRENT_TIMESTAMP
FROM users WHERE email = 'nilson.peres@usp.br'
ON CONFLICT DO NOTHING;

-- Insert test images and detections
WITH inserted_image AS (
    INSERT INTO images (source, cloud_platform, processing_time)
    VALUES ('https://example.com/test1.jpg', 'AWS', 0.18)
    RETURNING id
)
INSERT INTO detections (image_id, class, score, bounding_box)
SELECT 
    id, 'person', 0.95, '{"x": 100, "y": 200, "width": 50, "height": 100}'::jsonb
FROM inserted_image;

WITH inserted_image AS (
    INSERT INTO images (source, cloud_platform, processing_time)
    VALUES ('https://example.com/test2.jpg', 'Azure', 0.22)
    RETURNING id
)
INSERT INTO detections (image_id, class, score, bounding_box)
SELECT 
    id, 'car', 0.88, '{"x": 300, "y": 150, "width": 100, "height": 60}'::jsonb
FROM inserted_image;

-- Insert model metrics for the last 7 days
INSERT INTO model_metrics (model_id, date, inference_count, avg_inference_time, avg_confidence, error_count)
SELECT 
    m.id,
    CURRENT_DATE - (n || ' days')::interval,
    1000 + random() * 5000,
    0.15 + random() * 0.1,
    0.9 + random() * 0.1,
    random() * 50
FROM models m
CROSS JOIN generate_series(0, 6) n
WHERE m.name = 'YOLOv5'
ON CONFLICT (model_id, date) DO NOTHING;

INSERT INTO model_metrics (model_id, date, inference_count, avg_inference_time, avg_confidence, error_count)
SELECT 
    m.id,
    CURRENT_DATE - (n || ' days')::interval,
    800 + random() * 4000,
    0.2 + random() * 0.15,
    0.85 + random() * 0.15,
    random() * 40
FROM models m
CROSS JOIN generate_series(0, 6) n
WHERE m.name = 'Faster R-CNN'
ON CONFLICT (model_id, date) DO NOTHING;

-- Insert model metrics for nilson.peres@usp.br models
INSERT INTO model_metrics (model_id, date, inference_count, avg_inference_time, avg_confidence, error_count)
SELECT 
    m.id,
    CURRENT_DATE - (n || ' days')::interval,
    2000 + random() * 6000,
    0.12 + random() * 0.08,
    0.95 + random() * 0.05,
    random() * 30
FROM models m
CROSS JOIN generate_series(0, 6) n
WHERE m.name = 'YOLOv8'
ON CONFLICT (model_id, date) DO NOTHING;

INSERT INTO model_metrics (model_id, date, inference_count, avg_inference_time, avg_confidence, error_count)
SELECT 
    m.id,
    CURRENT_DATE - (n || ' days')::interval,
    1500 + random() * 5000,
    0.25 + random() * 0.15,
    0.92 + random() * 0.08,
    random() * 25
FROM models m
CROSS JOIN generate_series(0, 6) n
WHERE m.name = 'DETR'
ON CONFLICT (model_id, date) DO NOTHING;

-- Insert cloud metrics for the last 7 days
INSERT INTO cloud_metrics (platform, date, request_count, avg_latency, cost)
VALUES 
    ('AWS', CURRENT_DATE - '6 days'::interval, 5000, 0.18, 25.50),
    ('AWS', CURRENT_DATE - '5 days'::interval, 5500, 0.17, 28.75),
    ('AWS', CURRENT_DATE - '4 days'::interval, 6000, 0.19, 30.25),
    ('AWS', CURRENT_DATE - '3 days'::interval, 5800, 0.16, 29.50),
    ('AWS', CURRENT_DATE - '2 days'::interval, 6200, 0.20, 31.75),
    ('AWS', CURRENT_DATE - '1 day'::interval, 6500, 0.18, 32.50),
    ('AWS', CURRENT_DATE, 7000, 0.17, 35.00),
    ('Azure', CURRENT_DATE - '6 days'::interval, 4500, 0.22, 22.50),
    ('Azure', CURRENT_DATE - '5 days'::interval, 4800, 0.21, 24.00),
    ('Azure', CURRENT_DATE - '4 days'::interval, 5000, 0.23, 25.00),
    ('Azure', CURRENT_DATE - '3 days'::interval, 5200, 0.20, 26.00),
    ('Azure', CURRENT_DATE - '2 days'::interval, 5500, 0.22, 27.50),
    ('Azure', CURRENT_DATE - '1 day'::interval, 5800, 0.21, 29.00),
    ('Azure', CURRENT_DATE, 6000, 0.19, 30.00)
ON CONFLICT (platform, date) DO NOTHING;