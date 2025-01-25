export interface Prediction {
    model: string
    predictions: {
        label: string
        probability: number
    }[]
} 

export interface YoloImagePrediction {
    class_counts: Record<string,number>,
    confidence_scores: number[],
    boxes: number[][],
    processed_image: string
}

