"use client";

import { Atom, Camera, Guitar, Headset, ImageUp, Microscope, RotateCcw, Sparkle, SwitchCamera } from 'lucide-react'
import React, { useState } from 'react'
import { Button } from '../ui/button';
import PredictionLoadingSkeleton from './PredictionLoadingSkeleton';
import { Prediction } from '@/globals';
import { makeImagePredictionRequest, makeYoloImagePredictionRequest, makeYoloVideoPredictionRequest } from '@/lib/api';
interface props {
    modelId: string
}

function base64ToFile(base64:string, fileName="image.png", mimeType = "image/png") {
    const bstr = atob(base64); // Decode base64
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }

    return new File([u8arr], fileName, { type: mimeType });
}

const ImageUploadView:React.FC<props> = ({modelId}) => {
    const [file,setFile] = useState<File|null>(null)
    const [vidUrl,setVidUrl] = useState<string|null>(null)
    const [isPredicting,setIsPredicting] = useState(false)
    const [predictions,setPredictions] = useState<string[][]>([])
    
    
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => { 
        setVidUrl(null)
        setIsPredicting(false)
        setPredictions([])
        const _file = event.target.files?.[0]
        if(_file) setFile(_file)
        event.target.value = '';
    }

    const handlePredict = async () => { 
        if(!file) return
        setIsPredicting(true)
        const formData = new FormData()
        
        formData.append("file",file)
        formData.append("model",modelId)

        try {
            if(file.type.startsWith('video/')){
                formData.append("confidence", "0.2");
                const data = await makeYoloVideoPredictionRequest(formData)
                // console.log(data)
                setVidUrl(data)
            }
            else if(modelId != "yolo"){
                const data = await makeImagePredictionRequest(formData)
                setPredictions(data)
            }else{
                const data = await makeYoloImagePredictionRequest(formData)
                const _file = base64ToFile(data.img)
                setPredictions(data.answer)
                setFile(_file)
            }

            setIsPredicting(false)
        } catch (error) {
            
        }
    }
    

    return (
        <div className="flex justify-center items-center gap-6 flex-col">
            <input onChange={handleFileChange} id='fileInput' hidden type="file" accept="image/*,video/*" />
            {
                file ? 
                    <>
                        <div className={`${isPredicting ? "hidden" : ""}`}>
                            {
                                file.type.startsWith('video/') ? 
                                <video src={vidUrl ? vidUrl : URL.createObjectURL(file)} className="h-[400px] w-auto" controls autoPlay />
                                :
                                <img src={URL.createObjectURL(file)} className="h-[400px] w-auto" />
                            }
                        </div>
                        <PredictionLoadingSkeleton isPredicting={isPredicting} />
                        {
                            predictions.length > 0 ?
                                <div className='flex flex-wrap gap-x-10 gap-y-4 justify-between'>
                                    {predictions.map(_prediction=>(
                                        <div className='w-fit flex justify-center items-center gap-6'>
                                            <div className='flex justify-center items-center gap-2'>
                                                <Atom className='text-slate-600 h-5 w-auto' />
                                                <span>{_prediction[0]}</span>
                                            </div>
                                            <div className='flex justify-center items-center gap-2'>
                                                <Microscope className='text-slate-600 h-5 w-auto' />
                                                <span>{_prediction[1]}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            : ""
                        }
                        <div className='flex justify-center items-center gap-2'>
                            <Button>
                                <label htmlFor='fileInput' className='flex justify-center items-center gap-2 cursor-pointer'>
                                    <RotateCcw />
                                    <span>Retake</span>
                                </label>
                            </Button>
                            <Button onClick={handlePredict}>
                                <Sparkle />
                                <span>Predict</span>
                            </Button>
                        </div>
                    </>
                :
                <>
                    <label htmlFor='fileInput' className="cursor-pointer flex justify-center items-center opacity-70 hover:opacity-100 hover:border-solid group transition-all gap-4 border-2 border-dashed rounded-md min-h-[200px] w-full">
                        <ImageUp className="h-6 w-auto group-hover:-translate-y-3 transition-all" />
                        <span className="font-bold group-hover:-translate-y-3 transition-all">Upload Image or Video</span>
                    </label>
                </>
            }
        </div>
    )
}

export default ImageUploadView