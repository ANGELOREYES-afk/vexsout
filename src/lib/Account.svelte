<script lang="ts">
  import {Jumper} from 'svelte-loading-spinners';
  import { getContext, onMount, setContext} from 'svelte'
  import type { AuthSession } from '@supabase/supabase-js'
  import { supabase } from '../supabaseClient'
  import { RobotStore, type Robot } from '../store';
  import { writable } from 'svelte/store';
  import Auth from './Auth.svelte';

  export let session: AuthSession

  let loading = false
  let store = new RobotStore().robots;
  let teamname: string | null = null;
  let drivespeed: number | null = null;
  let wheelsize: number | null = null;
  let image_name: string | null = null;
  let intaketype: string | null = null;
  let intakespeed: number | null = null;
  let tierhang: string | null = null;
  let robotsize: number[] = [0, 0];
  let robotid: string | null = null;
  
  onMount(() => {
    getRobots()
  })

  const getRobots = async () => {
    try {
      loading = true
      const { user } = session

      const { data, error, status } = await supabase
        .from('robots')
        .select('*')
        .eq('user_id', user.id) 

      if (error && status !== 406) throw error

      if (data) { 
        for(const robot of data){
           store.push({teamname: robot.teamname, drivespeed: robot.drivespeed, uuid: robot.robot_id, user_uuid: user.id, update_at: robot.updated_at, image_name: robot.image_name, wheelsize: robot.wheelsize, intaketype: robot.intaketype, tierhang: robot.tierhang, robotsize: robot.robotsize, intakespeed: robot.speedintake}) 
        }
        return store;
      }
    } catch (error) {
      if (error instanceof Error) {
        alert(error.message)
        return [error.message, 0]
      }
    } finally {
      loading = false
    }
}
const updatesupabase = async () => { 
 try{
    loading = true 
     const { error } = await supabase
     .from('robots')
     .upsert({
        id: robotid, 
        updated_at: new Date(),
        teamname: teamname, 
        drivespeed: drivespeed,
        image_name: image_name,
        wheelsize: wheelsize,
        robotsize: robotsize,
        intaketype: intaketype,
        speedintake: intakespeed,
        tierhang: tierhang
     })

      if (error) {
        throw error
      }
    } catch (error) {
      if (error instanceof Error) {
        alert(error.message)
      }
    } finally {
      loading = false
    }
} 
const updateProfile = async () => {
    const data = RobotStore.robots.find((robot: Robot) => robot.teamname == teamname);
    robotid = data.uuid
    teamname = data.teamname 
    drivespeed = data.drivespeed 
    // uuid: data.uuid,
    // user_uuid: user.id,
    image_name =  data.image_name
    wheelsize =  data.wheelsize  
    intaketype = data.intaketype
    intakespeed = data.intakespeed
    tierhang =  data.tierhang 
    robotsize =  data.robotsize 
      // updated_at: new Date().toISOString(),
      // let list_1 = [teamname, drivespeed, image_name, wheelsize, intaketype, intakespeed, tierhang, robotsize];
 }

  let button_in_progress = 'gallery_button';

  let update_button =false; 
  // $: update_button.set(false) 
  // setContext('update_button', update_button); 
  let form_button = false; 
  // $: form_button.set(true);
  // setContext('form_button', form_button); 
  let gallery_button = true; 
  // $: gallery_button.set(false);
  // setContext('gallery_button', gallery_button);

  const button_switch = async (event: any) =>{
    loading = true;
    button_in_progress = event.target.value;

    robotid = null 
    teamname = null 
    drivespeed = null
    image_name = null 
    wheelsize = null 
    intaketype = null 
    intakespeed = null
    tierhang = null 
    robotsize =  [0, 0];
    
    update_button = false; 
    gallery_button = false;
    form_button = false;  
    if(button_in_progress == "update_button"){
      update_button = true; 
    }
    if(button_in_progress == "gallery_button"){
      gallery_button = true;
    }
    if(button_in_progress == "form_button"){
      form_button = true;
    }
    loading = false;
    console.log(update_button)
    console.log(gallery_button)
    console.log(form_button)
  }
  let fileinput: any;
  let fileimagename: any; 
const handleuploadimage = async (event: any)=>{
    loading = true;
    try{
       const {user} = session;
       const {data, error}  = await supabase.storage
        .from('robotimages')
        .upload(`${user.id}/${image_name}`, event.target.files[0])
        console.log('error happened')
        if(error){
          throw error 
          console.log("error happened")
        } 
        let image = event.target.files[0];
            // let reader: any | null;
            // reader = new FileReader();
            // reader.readAsDataURL(image);
            // reader.onload = (e: { target: {result: any; }; }) => {
            //     fileimagename = !e.target.result
            //     console.log(fileimagename)

            // };
        fileimagename=URL.createObjectURL(image);
      } catch (error) {
      if (error instanceof Error) {
        alert(error.message)
      }
    } finally {
      loading = false
    }
}
  const insertsupabase = async () => {
    try{ 
    const { user } = session;
    const {data, error} = await supabase 
      .from('robots')
      .insert({
          user_id: user.id,
          robot_id: crypto.randomUUID(), 
          updated_at: new Date(), 
          teamname: teamname, 
          drivespeed: drivespeed,
          image_name: image_name,
          wheelsize: wheelsize,
          robotsize: robotsize,
          intaketype: intaketype,
          speedintake: intakespeed,
          tierhang: tierhang
    })
      loading = false
      store = new RobotStore().robots;
      teamname = null;
      drivespeed = null;
      wheelsize = null;
      image_name = null;
      intaketype = null;
      intakespeed = null;
      tierhang = null;
      robotsize = [0, 0];
      robotid = null;

      console.log("successful")
      
    }catch (error) {
      if (error instanceof Error) {
        alert(error.message)
        return [error.message, 0]
      }
      } finally {
        loading = false
      }
  }
</script>

<!-- header : <button type="button" class="button block" on:click={() => supabase.auth.signOut()}> Sign Out-->
<!-- ok so heres update now since we can get session.UserActivation.email or  
sesion.UserActivation.uuid and in place of user_id in th-->
<!-- svelte-ignore empty-block -->
{#if loading}
<Jumper size="60" color="#FF3E00" unit="px" duration="1s"/>


{:else}
<button value="form_button" id="form_change" on:click={e => button_switch(e)}>Form</button>
<button value="update_button" id="update_change"on:click={e => button_switch(e)}>Update</button>
<button value="gallery_button" id="gallery_change"on:click={e => button_switch(e)}>Gallery</button>

<!-- svelte-ignore empty-block -->
{#if form_button}
 
<!-- {/if} -->

<!-- {#if getContext('form_button') == true} -->

<form class="form-widget">
  <div>sector: {session.user.email}</div>
  <div>
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
      <img class="upload" src="https://static.thenounproject.com/png/625182-200.png" alt="" on:click={()=>{fileinput.click();}} />
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <div class="chan" on:click={()=>{fileinput.click();}}>Choose Image</div>
      <input style="display:none" type="file" accept=".jpg, .jpeg, .png" on:change={(e)=>handleuploadimage(e)} bind:this={fileinput}> 
  </div>
  {#if fileimagename}
    <!-- svelte-ignore a11y-missing-attribute -->
    <img src="{fileimagename}" class="imageprop">
    <p>hello this is fileinput(just uploaded)</p>
  {/if}
  <div>
    <label for="teamname">teamname</label>
    <input id="teamname" type="text" bind:value="{teamname}" />
  </div>
  <div>
    <label for="username">Image_name</label>
    <input id="image_name" type="text" bind:value={image_name} /> 
  </div>
  <div>
    <label for="drivespeed">Drivespeed</label>
    <select bind:value={drivespeed} name="drivespeed" id="drivespeed">
      <option value="257">
       257
      </option>
      <option value="360">
        360
      </option>
      <option value="400">
       400 
      </option>
      <option value="450">
       450
      </option>
      <option value="600">
       600 
      </option>
    </select>
  </div>
  <div>
    <label for="wheelnumbers">WheelSizes</label>
    <select bind:value={wheelsize} name="wheelsize" id="wheelsize">
      <option value="2">
       2
      </option>
      <option value="2">
       2.75 
      </option>
      <option value="3.25">
       3.25
      </option>
      <option value="4">
       4
      </option>
    </select>
  </div>
  <div>
    <label for="intaketype">intaketype</label>
    <select bind:value={intaketype} name="intaketype" id="intaketype">
      <option value="banded">
        <button>banded</button>
      </option>
      <option value="flex wheel">
        <button>flex wheel</button>
      </option>
    </select>
  </div>
  <div class="flex">
    <label for="tierhang">intakespeed</label>
    <input bind:value={intakespeed} type="text" id="intakespeed"/><p>rpm</p>
  </div>
  <div>
    <label for="tierhang">tierhang</label>
    <select bind:value={tierhang} name="tierhang" id="tierhang">
      <option value="a">
       a
      </option>
      <option value="b">
       b 
      </option>
      <option value="c">
       c 
      </option>
      <option value="d">
       d
      </option>
      <option value="e">
       e
      </option>
      <option value="f">
       f
      </option>
      <option value="g">
       g
      </option>
      <option value="h">
       h
      </option>
    </select>
  </div>
  <div>
    <label for="website">robotsize</label>
    <hr>
    <input id="robotsizex" type="text" bind:value="{robotsize[0]}"/><p>x-bars?</p>
    <input id="robotsizey" type="text" bind:value={robotsize[1]}/><p>y-bars?</p>
  </div> 
  <div>
    <button type="submit" class="button primary block" on:click={() => insertsupabase()} disabled="{loading}">
      {loading ? 'Saving ...' : 'Save Changes'}
    </button>
  </div>
</form>

{/if}
{#if gallery_button}
 {#each store as sto} 
    <h1>teamname: {sto.teamname}</h1>
    <h3>drivespeed: {sto.drivespeed} rmp</h3>
    <h3>wheelsize: {sto.wheelsize} inches</h3>
    <h3>intakespeed: {sto.intakespeed} rpm</h3>
    <h3>intaketype: {sto.intaketype}</h3>
    <h3>robotsize: {sto.robotsize[0]} 'x' by {sto.robotsize[1]} 'y'</h3> 
  {/each}
{/if}
{#if update_button}
<form on:submit|preventDefault={updateProfile} class="form-widget">

  <div>sector: {session.user.email}</div>
  <div>
    <label for="teamname">teamname</label>
    <input id="teamname" type="text" bind:value="{teamname}" />
    <button on:submit={updateProfile}>find team stats</button>
  </div>

  <div>
    <label for="username"></label>
    <input id="image_name" type="text" bind:value={drivespeed} /> 
  </div>
  <div>
    <label for="drivespeed">Drivespeed</label>
    <select bind:value={drivespeed} name="drivespeed" id="drivespeed">
      <option value="257">
       257
      </option>
      <option value="360">
        360
      </option>
      <option value="400">
       400 
      </option>
      <option value="450">
       450
      </option>
      <option value="600">
       600 
      </option>
    </select>
  </div>
  <div>
    <label for="wheelnumbers">WheelSizes</label>
    <select bind:value={wheelsize} name="wheelsize" id="wheelsize">
      <option value="2">
       2
      </option>
      <option value="2">
       2.75 
      </option>
      <option value="3.25">
       3.25
      </option>
      <option value="4">
       4
      </option>
    </select>
  </div>
  <div>
    <label for="intaketype">intaketype</label>
    <select bind:value={intaketype} name="intaketype" id="intaketype">
      <option value="banded">
        <button>banded</button>
      </option>
      <option value="flex wheel">
        <button>flex wheel</button>
      </option>
    </select>
  </div>
  <div class="flex">
    <input bind:value={intakespeed} type="text" id="intakespeed"/><p>rpm</p>
  </div>
  <div>
    <label for="tierhang">tierhang</label>
    <select bind:value={tierhang} name="tierhang" id="tierhang">
      <option value="a">
       a
      </option>
      <option value="b">
       b 
      </option>
      <option value="c">
       c 
      </option>
      <option value="d">
       d
      </option>
      <option value="e">
       e
      </option>
      <option value="f">
       f
      </option>
      <option value="g">
       g
      </option>
      <option value="h">
       h
      </option>
    </select>
  </div>
  <div>
    <label for="website">robotsize</label>
    <hr>
    <input id="robotsizex" type="text" bind:value="{robotsize[0]}"/><p>x-bars?</p>
    <input id="robotsizey" type="text" bind:value={robotsize[1]}/><p>y-bars?</p>
  </div> 
  <div>
    <button type="submit" class="button primary block" on:click={() => updatesupabase()} disabled="{loading}"> {loading ? 'Saving ...' : 'Save Changes'}
    </button>
  </div>  
</form>
{/if}
{/if}