import { Prediction, YoloImagePrediction } from "@/globals"
import axios from "axios"
import { sleep } from "./utils"

export const API_URL = "https://smartbin.anvituteja.in/api/v1"

export const api = axios.create({
    baseURL: API_URL,
    // withCredentials: true
})
export async function makeImagePredictionRequest(formData: FormData) {
    const resp = await api.post<Prediction>("/api/inference", formData)
    const data = resp.data.predictions
    return data.map(_ele=>[_ele.label.replace("Predicted: ","Category: "),`Accuracy: ${_ele.probability.toString()}%`])
}

export async function makeYoloImagePredictionRequest(formData: FormData) {
    const resp = await api.post<YoloImagePrediction>("/detect/yolo/image", formData)
    return {
        answer: Object.entries(resp.data.class_counts).map(([key,value]) => [`Category: ${key}`,`Count: ${value.toString()}`]),
        img: resp.data.processed_image
    }
}

export async function makeYoloVideoPredictionRequest(formData: FormData):Promise<string> {
    const resp = await api.post<string>("/detect/yolo/video", formData)
    return `${API_URL}/yolo_video/${resp.data}`
    // return `${process.env.API_URL}/yolo_video/${resp.data.split("/")[1]}`
}