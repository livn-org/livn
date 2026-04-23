<script lang="ts">
    let { activeTab, onTabChange }: {
        activeTab: 'bio' | 'sim' | 'build';
        onTabChange: (tab: 'bio' | 'sim' | 'build') => void;
    } = $props();
</script>

<nav class="nav-bar">
    <div class="brand">livn</div>
    <button class="tab" class:active={activeTab === 'bio'} onclick={() => onTabChange('bio')}>Bio</button>
    <button class="tab" class:active={activeTab === 'sim'} onclick={() => onTabChange('sim')}>Sim</button>
    <button class="tab" class:active={activeTab === 'build'} onclick={() => onTabChange('build')}>Build</button>
</nav>

<style>
    .nav-bar {
        display: flex;
        align-items: center;
        background: #0d0d1a;
        border-bottom: 1px solid #2a2a4a;
        padding: 0 16px;
        height: 38px;
        flex-shrink: 0;
    }
    .brand {
        font-size: 13px;
        font-weight: 700;
        color: #4fc3f7;
        letter-spacing: 0.1em;
        margin-right: 20px;
    }
    .tab {
        background: none;
        border: none;
        border-bottom: 2px solid transparent;
        color: #777;
        font-size: 13px;
        font-weight: 600;
        padding: 0 14px;
        height: 100%;
        cursor: pointer;
        transition: color 0.15s, border-color 0.15s;
    }
    .tab:hover {
        color: #bbb;
    }
    .tab.active {
        color: #4fc3f7;
        border-bottom-color: #4fc3f7;
    }
</style>
