import { createClient } from '@supabase/supabase-js'

const supabaseUrl = "https://txgusoicnbjmmkyjtjrs.supabase.co" 
const supabaseAnonKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR4Z3Vzb2ljbmJqbW1reWp0anJzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTAyMDgxNzcsImV4cCI6MjAyNTc4NDE3N30.rcsv-2sp05CYUvwouxc3Unbm65XpEjrmrncDD_RTsoQ"


export const supabase = createClient(supabaseUrl, supabaseAnonKey)
