#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "Game.hpp"

Game::Game(int numPlayers, sf::RenderWindow &window) {
    for (int i = 0; i < numPlayers; ++i) {
        foods.push_back(Food(*(new sf::Vector2f(window.getSize().x, window.getSize().y))));
    }
    basket = new Basket(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y)));
}

void Game::draw(sf::RenderWindow &window) {
    basket->draw(window);
    for (GameObject &g : foods) {
        g.draw(window);
    }
}

void Game::setFood(int food) {
    if (food < foods.size()) {
        currentFood = food;
    }
}

void Game::tick() {
}