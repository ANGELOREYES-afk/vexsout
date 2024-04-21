<!-- gmail: create(1) robotcreate01@gmail.com password: abcdefghijk!-->
<!-- gmail: test(2)  robottest46@gmail.com password: abcdefghijk!-->
<script lang="ts">
  import { onMount } from 'svelte'
  import { supabase } from './supabaseClient'
  import type { AuthSession } from '@supabase/supabase-js'
  import Account from './lib/Account.svelte'
  import Auth from './lib/Auth.svelte'
  let session: AuthSession | null;

  onMount(() => {
    supabase.auth.getSession().then(({ data }) => {
      session = data.session
    })

    supabase.auth.onAuthStateChange((_event, _session) => {
      session = _session
    })
  })

</script>

<main>
  <div class="container" style="padding: 50px 0 100px 0">
    <div>
      hello
    </div>
    {#if !session}
    <Auth />
    {:else}
    <Account {session} />
    {/if}
  </div>  
</main>
