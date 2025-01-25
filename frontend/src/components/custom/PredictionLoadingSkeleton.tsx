import React from 'react'
import { Skeleton } from '../ui/skeleton'
import { Image, Video } from 'lucide-react'
interface props {
    isPredicting: boolean
}

const PredictionLoadingSkeleton:React.FC<props> = ({isPredicting}) => {
  return (
    <div className={`flex flex-col gap-1 w-full ${isPredicting ? "block" : "hidden"}`}>
        <Skeleton className="rounded h-[400px] w-full mx-auto flex justify-center items-center flex-col gap-1">
            <div className='text-slate-600 flex justify-center items-center gap-2'>
                <Image />
                <Video />
            </div>
            <span className='font-lighter '>Your image/video is being processed, Please wait</span>
        </Skeleton>
        <div className='w-full flex justify-center items-center gap-4'>
            <Skeleton className="rounded h-[30px] w-[200px]" />
            <Skeleton className="rounded h-[30px] w-[200px]" />
        </div>      
    </div>
  )
}

export default PredictionLoadingSkeleton