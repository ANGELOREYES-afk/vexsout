<script lang="ts">
    let loading = false;
    let fileinput: any;
    let fileimagename: any; 

    const handleuploadimage = async (event: any)=>{
    loading = true;
    try{
       const {user} = session;
       const {data, error}  = await supabase.storage
        .from('robotimages')
        .upload(user.id + "/" + image_name, event.target.files[0])
        if(error){
          throw error 
        } 
        let image = event.target.files[0];
            let reader: any | null;
            reader = new FileReader();
            reader.readAsDataURL(image);
            reader.onload = (e: { target: { result: any; }; }) => {
                fileimagename = !e.target.result
            };
      } catch (error) {
      if (error instanceof Error) {
        alert(error.message)
      }
    } finally {
      loading = false
    }
}

</script>

<div>
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
      <img class="upload" src="https://static.thenounproject.com/png/625182-200.png" alt="" on:click={()=>{fileinput.click();}} />
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <div class="chan" on:click={()=>{fileinput.click();}}>Choose Image</div>
      <input style="display:none" type="file" accept=".jpg, .jpeg, .png" on:change={(e)=>handleuploadimage(e)} bind:this={fileinput} >
    {#if fileinput}
      <!-- svelte-ignore a11y-missing-attribute -->
      <img src="{fileinput}" class="imageprop">
      <p>hello this is fileinput(just uploaded)</p>
    {/if}
</div>
