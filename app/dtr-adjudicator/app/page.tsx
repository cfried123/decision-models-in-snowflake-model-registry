import { Adjudicator } from "@/components/adjudicator"

// Required: Snowflake is not reachable during docker build.
export const dynamic = "force-dynamic"

export default function Home() {
  return (
    <main className="w-full">
      <Adjudicator />
    </main>
  )
}
