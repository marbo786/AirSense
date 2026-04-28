FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom config
COPY nginx.conf /etc/nginx/conf.d/

# Copy static frontend HTML/CSS/JS
COPY . /usr/share/nginx/html

EXPOSE 80
CMD sh -c "sed -i \"s/MAPBOX_TOKEN_PLACEHOLDER/$MAPBOX_TOKEN/g\" /usr/share/nginx/html/app.js && nginx -g 'daemon off;'"
