<!-- The theme's agenda layout with one fix: it read `$frontmatter.value`, which
     is undefined (useSlideContext returns $frontmatter unwrapped), so a slide's
     `title:` never reached the heading and every agenda read "Agenda". -->
<template>
  <div class="slidev-layout agenda h-full flex flex-col">
    <div class="flex-grow flex items-center justify-center">
      <div class="w-full max-w-3xl px-8">
        <h1 class="text-3xl font-bold mb-10 text-center text-primary tracking-tight">
          {{ $frontmatter.title || 'Agenda' }}
        </h1>
        <div class="space-y-4">
          <div v-for="(item, idx) in items || []" :key="idx"
               class="agenda-item flex items-center p-4 bg-white rounded-lg shadow-sm border border-gray-100">
            <div class="flex-shrink-0 w-10 h-10 flex items-center justify-center rounded-full bg-primary text-white font-bold text-lg mr-5">
              {{ idx + 1 }}
            </div>
            <div class="agenda-text text-xl font-serif text-gray-800">{{ item }}</div>
          </div>
        </div>
        <div class="mt-8">
          <slot />
        </div>
      </div>
    </div>
    <ScholarlyFooter class="flex-shrink-0" />
  </div>
</template>

<script setup lang="ts">
import { useSlideContext } from '@slidev/client'

defineProps<{ items?: string[] }>()
const { $frontmatter } = useSlideContext()
</script>

<style scoped>
.text-primary { color: var(--slidev-theme-primary); }
.bg-primary { background-color: var(--slidev-theme-primary); }
</style>
