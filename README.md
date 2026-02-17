what did this fork do?

changed prompts to use automatic1111 syntax including prompt editing
changed vae decoding to being single tile, blend on cpu - mostly so that 6700xt doesnt crash

invoke on 6700xt, with the same steps and resolution compared to reforge is still 3x slower

you probably want to turn cfg scaling to 0.2 to make it look the same
also be aware some models need manual editing to vpred
