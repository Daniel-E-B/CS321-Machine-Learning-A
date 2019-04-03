#include "Basket.hpp"
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"

Basket::Basket(sf::Vector2f &pos) {
    shape = sf::RectangleShape(*(new sf::Vector2f(SIZE, SIZE)));
    shape.setOrigin(SIZE / 2, SIZE / 2);
    shape.setFillColor(COLOR);
    shape.setPosition(pos);

    net << tiny_dnn::layers::fc(2, 5) << tiny_dnn::activation::leaky_relu()
        << tiny_dnn::layers::fc(5, 5) << tiny_dnn::activation::leaky_relu()
        << tiny_dnn::layers::fc(5, 2) << tiny_dnn::activation::tanh();  // TODO: one or two recurrents?
}

void Basket::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Basket::move(sf::Vector2f const &targetPos) { // TODO: make more things const when they don't need to change
    tiny_dnn::vec_t pred = net.predict({targetPos.x, targetPos.y});
    shape.move(*(new sf::Vector2f(pred[0], pred[1])));
}

sf::Vector2f Basket::getPos() {
    return shape.getPosition();
}

std::string Basket::getBrain(){
    return net.to_json();
}

void Basket::setBrain(std::string &brain){
    net.from_json(brain);
}