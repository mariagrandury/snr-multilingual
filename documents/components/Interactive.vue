<!-- An interactive figure of the project site (public/embed.html), e.g.
     <Interactive viz="paperRq1" panel="a" />. The URL follows the deck's base,
     so it works in dev, in presenter mode and in a build under /slides/. -->
<template>
  <iframe class="interactive" :src="src" :style="{ height }" loading="lazy" />
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ viz: string; panel?: string; zoom?: number; height?: string }>()

const src = computed(() => {
  const q = new URLSearchParams({ viz: props.viz })
  if (props.panel) q.set('panel', props.panel)
  if (props.zoom) q.set('zoom', String(props.zoom))
  return `${import.meta.env.BASE_URL}embed.html?${q}`
})
const height = computed(() => props.height ?? '390px')
</script>

<style scoped>
.interactive { display: block; width: 900px; max-width: 100%; margin: 0 auto; border: 1px solid #fff; border-radius: 6px; background: #fff; box-shadow: none; }
</style>
