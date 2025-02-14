import discord
from discord import app_commands
import interphase
import logging
import warnings
import re

# Create a Discord client with the default intents
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)  # Slash command manager

#hide warnings
logging.getLogger("discord").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

api_url = "<your URL>"
model= "<your Model>"
db_name = "<your DB Name>"

embed_url = "<your URL2>"
embed_model = "<your embed Model>"

memory_of_convo = []


@client.event
async def on_ready():
    await tree.sync()
    print(f'Logged in as {client.user}!')  # Log when the bot is ready


# Define a slash command left for testing
@tree.command(name="ping", description="Replies with Pong!")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")


@tree.command(name="clear_memory", description="Clears The Bots Memory")
async def ping(interaction: discord.Interaction):
    global memory_of_convo
    memory_of_convo = []
    await interaction.response.send_message("Memory Cleared.")


@tree.command(name="listbooks", description="Replies with List of items in the database!")
async def ping(interaction: discord.Interaction):
    file_tuple_list=interphase.SearchDataEmbed.database_commands.list_books(db_name)
    file_names = [item[1] for item in file_tuple_list]
    await interaction.response.send_message("```"+"\n".join(file_names) + "\n```")


@tree.command(name="query", description="Ask the bot a question")
async def query(interaction: discord.Interaction, user_query: str):
    global memory_of_convo

    if user_query:
        await interaction.response.defer(thinking=True)  # Defer response to indicate processing

        # Retrieve relevant context from the database
        rag_item = interphase.SearchDataEmbed.search_and_return_results(db_name, user_query, embed_url, embed_model, 3)

                # Ensure memory stays within token limits
        while interphase.count_tokens(str(memory_of_convo)) > 131072: #you will need to update this to the model your usings context window
            memory_of_convo.pop(0)  # Forget the oldest memory entry

        # Generate AI response with current memory
        response, memory_of_convo = interphase.query_ai_system(api_url, user_query, model, rag_item, memory_of_convo)

        try:
            # Craft full message before splitting
            full_message = f"User Query:\n\n{user_query}\n\nResponse:\n\n{response}"

            # Function to split text while keeping sentence boundaries within Discord's limits
            def split_text(text, limit=1999):
                sentences = re.split(r'(?<=[.!?])\s+', text)  # Split at sentence boundaries
                chunks = []
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > limit:  # +1 for spacing
                        chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence

                if current_chunk:
                    chunks.append(current_chunk)

                return chunks

            messages = split_text(full_message)

            # Send first chunk as the initial response
            await interaction.edit_original_response(content=messages[0])

            # Send remaining chunks as follow-ups
            for msg in messages[1:]:
                await interaction.followup.send(msg)

        except Exception as e:
            await interaction.edit_original_response(content=f"Error: {str(e)}")






client.run('<your Bot Token>')
