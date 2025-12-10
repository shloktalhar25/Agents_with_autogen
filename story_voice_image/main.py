import os
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import requests
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
import asyncio


from elevenlabs.client import ElevenLabs
from elevenlabs.play import play



load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# creating the open router model client

open_router_model_client = OpenAIChatCompletionClient(
    model="openai/gpt-oss-20b", 
    api_key= OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model_info = {
    "type": "chat_completion",     
    "vision": False,               
    "audio": False,                
    "function_calling": False,
    "json_output": True,      
    "input_format": "text",        
    "output_format": "text",       
    "context_window": 8192,        
    "provider": "openrouter",      
    "description": "Open-source 20B chat model via OpenRouter",  
    "family": "gpt-oss",           
    "temperature": 1.0,            
    "top_p": 1.0,                  
}
)


# creating elevenlabs_client
tts_client = ElevenLabs(api_key = ELEVENLABS_API_KEY)


#---------------------------------------------------
#Creating tools and functins

# Adding tools and api integration

#3.1 attaching a text to speech api (used in voice agent)


def generate_voiceovers(messages: list[str]) -> list[str]:
    """
    Generate voiceovers for a list of messages using elevenlabs api

    Args:
        messages: List of message to convert to speech
    returns:
        List of file paths to the generated voiceover audio files
    """

    os.makedirs("voiceovers",exist_ok=True)

    # check for existing file first

    audio_file_path = [] 
    
    for i in range(1,len(messages)+1):
        file_path = f"voiceovers/voiceover_{i}.mp3"
        if os.path.exists(file_path):
            audio_file_path.append(file_path)

    # if all files exists , return them

    if len(audio_file_path) == len(messages):
        print("ALl voice over files alreay exists,skipping generation..")
        return audio_file_path
    
    # generate missing file one by one 
    audio_file_path = []

    for i , messages in enumerate(messages,1):
        try:
            save_file_path = f"voiceovers/voiceover_{i}.mp3"
            if os.path.exists(save_file_path):
                print(f"File{save_file_path} already exists,skpping generation.")
                audio_file_path.append(save_file_path)
                continue

            print(f"generating voiceover {i}/{len(messages)}..")

            #generate audio with elevenLabs
            response = tts_client.text_to_speech.convert(
                text=messages,
                voice_id= "JBFqnCBsd6RMkjVDRZzb",
                model_id= "eleven_multilingual_v2",
                output_format="mp3_22050_32",
            )

            # collect audio chunks
            audio_chunks = []
            for chunks in response:
                if chunks:
                    audio_chunks.append(chunks)
            
            # save to file
            with open(save_file_path,"wb") as f:
                for chunk in audio_chunks:
                    f.write(chunk)

            print(f"Voiceover {i} generated successfully")
            audio_file_path.append(save_file_path)

        except Exception as e:
            print(f"Error generating voiceover for message: {messages}. Error: {e}")
            continue
        
    return audio_file_path





def generate_images(prompts: list[str]):
    """
    Generate images based on text prompts using Stability AI API.
    
    Args:
        prompts: List of text prompts to generate images from
    """
    seed = 42
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)

    # API config
    stability_api_url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }

    for i, prompt in enumerate(prompts, 1):
        print(f"Generating image {i}/{len(prompts)} for prompt: {prompt}")

        # Skip if image already exists
        image_path = os.path.join(output_dir, f"image_{i}.webp")
        if not os.path.exists(image_path):
            # Prepare request payload
            payload = {
                "prompt": (None, prompt),
                "output_format": (None, "webp"),
                "height": (None, "1920"),
                "width": (None, "1080"),
                "seed": (None, str(seed))
            }

            try:
                response = requests.post(stability_api_url, headers=headers, files=payload)
                if response.status_code == 200:
                    with open(image_path, "wb") as image_file:
                        image_file.write(response.content)
                    print(f"Image saved to {image_path}")
                else:
                    print(f"Error generating image {i}: {response.json()}")
            except Exception as e:
                print(f"Error generating image {i}: {e}")


def generate_video(captions: list[str]) -> list[str]:
    """
    Generate a short video by combining generated images and voiceovers.

    Args:
        captions: List of text captions used to create video scenes.

    Returns:
        List containing a single file path to the generated video.
    """

    print("Starting video generation...")
    os.makedirs("videos", exist_ok=True)

    video_path = "videos/final_video.mp4"

    # If video already exists, skip generation
    if os.path.exists(video_path):
        print("Video already exists, skipping generation...")
        return [video_path]

    # Directories where images and audio were saved
    image_dir = "images"
    audio_dir = "voiceovers"

    # Collect clips
    video_clips = []

    for i, caption in enumerate(captions, 1):

        image_path = f"{image_dir}/image_{i}.webp"
        audio_path = f"{audio_dir}/voiceover_{i}.mp3"

        # Sanity checks
        if not os.path.exists(image_path):
            print(f"Missing image file: {image_path}")
            continue

        if not os.path.exists(audio_path):
            print(f"Missing audio file: {audio_path}")
            continue

        print(f"Combining image {i} + voiceover {i}")

        # Load audio
        audio_clip = AudioFileClip(audio_path)

        # Create image clip for duration of voiceover
        image_clip = (
            ImageClip(image_path)
            .set_duration(audio_clip.duration)
            .set_audio(audio_clip)
            .resize((1080, 1920))  # portrait format
        )

        video_clips.append(image_clip)

    if not video_clips:
        print("Error: No clips generated. Video not created.")
        return []

    # Concatenate all clips
    print("Rendering final video...")
    final_video = concatenate_videoclips(video_clips, method="compose")

    # Export video
    final_video.write_videofile(
        video_path,
        fps=30,
        codec="libx264",
        audio_codec="aac"
    )

    print(f"Video generated successfully: {video_path}")
    return [video_path]








#-------------------------------------------------------


# defining our script writter agent
script_writter = AssistantAgent(
    name ="script_writer",
    description="This agent would create a script /captions",
    model_client=open_router_model_client,
    system_message='''
        You are a creative assistant tasked with writing a script for a short video. 
        The script should consist of captions designed to be displayed on-screen, with the following guidelines:
            1.	Each caption must be short and impactful (no more than 8 words) to avoid overwhelming the viewer.
            2.	The script should have exactly 5 captions, each representing a key moment in the story.
            3.	The flow of captions must feel natural, like a compelling voiceover guiding the viewer through the narrative.
            4.	Always start with a question or a statement that keeps the viewer wanting to know more.
            5.  You must also include the topic and takeaway in your response.
            6.  The caption values must ONLY include the captions, no additional meta data or information.

            Output your response in the following JSON format:
            {
                "topic": "topic",
                "takeaway": "takeaway",
                "captions": [
                    "caption1",
                    "caption2",
                    "caption3",
                    "caption4",
                    "caption5"
                ]
            }
    '''
)




# agent for voice agent
voice_actor = AssistantAgent(
    name="voice_actor",
    model_client=open_router_model_client,
    tools=[generate_voiceovers],
    system_message='''
        You are a helpful agent tasked with generating and saving voiceovers.
        Only respond with 'TERMINATE' once files are successfully saved locally.
    '''
)



# defining the graphic desining Agent

graphic_designer = AssistantAgent(
    name = "graphic_designer",
    description="This agent would make graphics like photos",
    model_client = open_router_model_client,
    tools = [generate_images],
    system_message='''
        You are a helpful agent tasked with generating and saving images for a short video.
        You are given a list of captions.
        You will convert each caption into an optimized prompt for the image generation tool.
        Your prompts must be concise and descriptive and maintain the same style and tone as the captions while ensuring continuity between the images.
        Your prompts must mention that the output images MUST be in: "Abstract Art Style / Ultra High Quality." (Include with each prompt)
        You will then use the prompts list to generate images for each provided caption.
        Only respond with 'TERMINATE' once the files are successfully saved locally.
    '''   
)

director = AssistantAgent(
    name="director",
    model_client=open_router_model_client,
    tools=[generate_video],
    system_message='''
        You are a helpful agent tasked with generating a short video.
        You are given a list of captions which you will use to create the short video.
        Remove any characters that are not alphanumeric or spaces from the captions.
        You will then use the captions list to generate a video.
        Only respond with 'TERMINATE' once the video is successfully generated and saved locally.
    '''
)





# Set up termination condition
termination = TextMentionTermination("TERMINATE")

# Create the autogen team
agent_team = RoundRobinGroupChat(
    [script_writter, voice_actor, graphic_designer, director],
    termination_condition=termination,
    max_turns=4
)

async def interactive_console():
    while True:
        user_input = input("Enter a message (type 'exit' to leave): ")

        if user_input.strip().lower() == "exit":
            print("Exiting console…")
            break

        # Run the team with your input
        stream = agent_team.run_stream(task=user_input)

        # Stream the output to console
        await Console(stream)


# Run the async console loop
asyncio.run(interactive_console())
