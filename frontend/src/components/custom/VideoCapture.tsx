"use client";

import React, { useEffect, useRef, useState } from 'react'
import { Button } from '../ui/button';
import { Aperture, Atom, Camera, Guitar, Headset, Image, LoaderCircle, LucideCamera, Microscope, Sparkle, SwitchCamera, Video } from 'lucide-react';
import { sleep } from '@/lib/utils';
import { Skeleton } from "@/components/ui/skeleton"
import PredictionLoadingSkeleton from './PredictionLoadingSkeleton';
import { Prediction } from '@/globals';
import { api } from '@/lib/api';

interface props {
    modelId: string
}

const VideoCapture:React.FC<props> = ({modelId}) => {
    const videoRef = useRef<HTMLVideoElement|null>(null)
    const canvasRef = useRef<HTMLCanvasElement|null>(null)
    const [capture,setCapture] = useState(false)
    const [isPredicting,setIsPredicting] = useState(false)
    const [prediction,setPrediction] = useState<string[]>([])

    useEffect(()=>{
        (async()=>{
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: {
                    facingMode:"environment"
                } })
                if (videoRef.current) {
                  videoRef.current.srcObject = stream
                  videoRef.current.play()
                }
            } catch (err) {
                console.error('Error accessing camera:', err)
            }
        })()
    },[])

    const handleCapture = ()=>{
        if (videoRef.current && canvasRef.current) {
            const _video = videoRef.current
            const _canvas = canvasRef.current
            _canvas.width = _video.videoWidth;
            _canvas.height = _video.videoHeight;

            _canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0)
            setCapture(true)
        }
    }
    
    const getCanvasBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error("Failed to convert canvas to Blob."));
                }
            }, "image/jpeg");
        });
    };

    const handleReCapture = () => { 
        setPrediction([])
        setIsPredicting(false)
        setCapture(false)
    }

    const handlePredict = async () => { 
        if(canvasRef.current == null) return
        setIsPredicting(true)
        try {
            const blob = await getCanvasBlob(canvasRef.current);
            const formData = new FormData()
            
            formData.append("file",new File([blob], "captured-photo.jpg", { type: "image/jpeg" }))
            formData.append("model",modelId)

            const resp = await api.post<Prediction>("/api/inference",formData)
            const data = resp.data.predictions[0]
            setPrediction([data.label.replace("Predicted: ",""),data.probability.toString()])

            setIsPredicting(false)
            
        } catch (error) {
            
        }
    }


    return (
        <div className=''>
            <div className={`${capture ? "block" : "hidden"}`}>
                <canvas hidden={isPredicting} ref={canvasRef} className='mx-auto h-[400px]' />
                {
                    prediction.length > 0 ?
                    <div className='mx-auto w-fit flex justify-center items-center gap-10 mt-2 mb-4'>
                        <div className='flex justify-center items-center gap-2'>
                            <Atom className='text-slate-600 h-5 w-auto' />
                            <span>Category: {prediction[0]}</span>
                        </div>
                        <div className='flex justify-center items-center gap-2'>
                            <Microscope className='text-slate-600 h-5 w-auto' />
                            <span>Accuracy: {prediction[1]}</span>
                        </div>
                    </div>
                    : ""
                }
                <PredictionLoadingSkeleton isPredicting={isPredicting} />
                
                
                
            </div>
            <video hidden={capture} ref={videoRef} className='h-[400px] w-full' />
            <div className='w-full flex justify-center items-center gap-4 mt-2'>
                {
                    capture ?
                        <>
                            <Button onClick={handleReCapture}>
                                <SwitchCamera />
                                <span>Retake</span>
                            </Button>
                            <Button onClick={handlePredict}>
                                <Sparkle />
                                <span>Predict</span>
                            </Button>
                        </>
                    :
                        <Button onClick={handleCapture}>
                            <Aperture />
                            <span>Capture</span>
                        </Button>
                }
            </div>
        </div>
    )
}

export default VideoCapture