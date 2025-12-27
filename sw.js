// DockView Service Worker - PWA Caching
const CACHE_NAME = 'dockview-v1';
const urlsToCache = [
  '/',
  '/index.html'
];

// Install event - cache resources
self.addEventListener('install', event => {
  console.log('[SW] Installing service worker...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[SW] Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[SW] Activating service worker...');
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  // Skip caching for API requests
  if (event.request.url.includes('/containers') ||
      event.request.url.includes('/volumes') ||
      event.request.url.includes('/network') ||
      event.request.url.includes('/widgets') ||
      event.request.method !== 'GET') {
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // Clone the request
        const fetchRequest = event.request.clone();

        return fetch(fetchRequest).then(response => {
          // Check if valid response
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }

          // Clone the response
          const responseToCache = response.clone();

          caches.open(CACHE_NAME)
            .then(cache => {
              cache.put(event.request, responseToCache);
            });

          return response;
        }).catch(() => {
          // Return offline page or cached content if available
          return caches.match('/index.html');
        });
      })
  );
});

// Background sync for offline actions
self.addEventListener('sync', event => {
  console.log('[SW] Background sync:', event.tag);
  if (event.tag === 'sync-container-actions') {
    event.waitUntil(syncContainerActions());
  }
});

async function syncContainerActions() {
  // Placeholder for syncing offline container actions
  console.log('[SW] Syncing container actions...');
}

// Push notifications
self.addEventListener('push', event => {
  console.log('[SW] Push received:', event);
  const options = {
    body: event.data ? event.data.text() : 'New notification from DockView',
    icon: '/manifest.json',
    badge: '/manifest.json',
    vibrate: [200, 100, 200],
    tag: 'dockview-notification',
    requireInteraction: false
  };

  event.waitUntil(
    self.registration.showNotification('DockView', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  console.log('[SW] Notification click received');
  event.notification.close();

  event.waitUntil(
    clients.openWindow('/')
  );
});
