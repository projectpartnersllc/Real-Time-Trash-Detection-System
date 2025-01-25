"use client";

import {Select,SelectContent,SelectItem,SelectTrigger,SelectValue} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {Card,CardContent,CardDescription,CardHeader,CardTitle} from "@/components/ui/card"
import ImageUploadView from "@/components/custom/ImageUpload";
import VideoCapture from "@/components/custom/VideoCapture";
import { ParticlesView } from "@/components/custom/ParticlesView";
import { useState } from "react";


export default function Home() {
  const [modelId,setModelId] = useState("mobilenet_v3_small") 
  
  
  return (
    <section className="flex flex-col h-[100svh] relative">
      <ParticlesView />
      <div className="z-[100] bg-white w-full flex justify-center items-center py-4">
        <span className="text-3xl font-bold">AI Categorizer</span>
      </div>
      <div className="z-[100] flex-1 flex justify-center items-center flex-wrap">
        <div className="w-1/2 flex justify-center items-center min-w-[400px] relative">
          <Card className="w-[300px] md:w-[400px] bg-white bg-opacity-5 backdrop-blur-md">
            <CardHeader>
              <CardTitle>Select AI/ML Model</CardTitle>
              <CardDescription>Models for detecting image</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-2">
                <Select defaultValue={modelId} onValueChange={(value)=>setModelId(value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="None" />
                  </SelectTrigger>
                  <SelectContent className="z-[100]">
                    <SelectItem value="mobilenet_v3_small">Mobilenet V3 Small</SelectItem>
                    <SelectItem value="mobilenet_v2">Mobilenet V2</SelectItem>
                    <SelectItem value="mobilenet_v2_trashbox">Mobilenet V2 Trashbox</SelectItem>
                    <SelectItem value="mobilenet_v3_large">Mobilenet V3 Large</SelectItem>
                    <SelectItem value="resnet152">Resnet 152</SelectItem>
                    <SelectItem value="yolo">Yolo</SelectItem>
                  </SelectContent>
                </Select>
                <span className="text-sm text-slate-600">
                Explore our powerful AI/ML models to process your images or videos with precision. Make your selection to get started.
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="md:w-1/2 md:pr-10 min-w-[300px] md:min-w-[600px] relative">
          <Card className="mx-auto bg-white bg-opacity-5 backdrop-blur-md">
            <CardContent className="py-6 px-4">
              <Tabs defaultValue="upload" className="w-full">
                <TabsList className="w-full h-fit mb-4">
                  <TabsTrigger value="upload" className="flex-1 py-3 whitespace-break-spaces">Upload Photo/Video</TabsTrigger>
                  <TabsTrigger value="capture" className="flex-1 py-3 whitespace-break-spaces">Capture Photo</TabsTrigger>
                </TabsList>
                <TabsContent value="upload">
                  <ImageUploadView modelId={modelId} />
                </TabsContent>
                <TabsContent value="capture">
                  <VideoCapture modelId={modelId} />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
          
        </div>
      </div>
    </section>
  );
}
